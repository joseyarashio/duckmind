"""Layering (physical-design spec §2.2 rule R1, §17 phase 0): the mind does not reach into the
world, and the world does not reach into the mind.

* ``duckmind.duckcore`` imports nothing from the simulator page (``simweb``), the world
  (``world``), the duck's eyes (``eye``), the live physics bridge (``livebody``), the body
  host (``bodyhost``), the other simulators and the simulator oracle (``simulator``,
  ``sim_live``, ``sim_pet``, ``local_env``) or physics (``mujoco``, the official
  ``mjlab_microduck``): neither directly nor through any duckmind module it imports.
* ``duckmind.world`` and ``duckmind.eye`` use no Duck Core internals: at most its integer
  maths (``fixed``) and its counter-based draws (``rng``). The percept contract is
  ``duckmind.percept`` (phase 3), outside Duck Core.
* The board side — duck-body (``duckmind.duckbody``), duck-eye (``duckmind.eye``), the curiosity
  Act Runner (``duckmind.curiosity``) and the percept contract (``duckmind.percept``) — runs on
  the real duck: nothing of the simulator, the physics, the page or the world, and nothing of
  Duck Core at all (importing even ``duckmind.duckcore.fixed`` runs the whole package's
  ``__init__``; duck-body keeps its own copy of the integer trig, ``duckbody/fixedmath.py``).
* Phase 4 (「邊界」): the world port ``duckmind.worldport`` is neutral — the standard library and
  ``duckmind.percept`` only — so Duck Core, the worlds, the api and the tools can all import it;
  every engine revision (``duckmind.engines``: the frozen v4 engine and the migration seam, which
  loads ``duckmind.duckcore.api``) keeps rule R1; LegacyWorld (``duckmind.world.legacy``) uses the
  port, percept-v1 and Duck Core's public maths only; and the board side loads neither the world
  port nor an engine (they are the simulator's and Duck Core's side of the boundary).
* ``duckmind.simweb`` may import anything.

Imports are read with ``ast``; nothing is imported. Module-level and function-level imports,
relative imports, ``if TYPE_CHECKING:`` imports, and ``importlib.import_module("…")`` /
``__import__("…")`` with a literal name all count.
"""

from __future__ import annotations

import ast
import textwrap
from functools import cache
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
DUCKCORE = "duckmind.duckcore"
NOT_FOR_THE_MIND = (
    "duckmind.simweb",
    "duckmind.world",
    "duckmind.eye",
    "duckmind.livebody",
    "duckmind.bodyhost",
    "duckmind.simulator",
    "duckmind.sim_live",
    "duckmind.sim_pet",
    "duckmind.local_env",
    "mujoco",
    "mjlab_microduck",
)
SENSE_SIDE = ("duckmind.world", "duckmind.eye")
# The board side (phase 3: duck-body, duck-eye, the curiosity Act Runner, percept-v1) runs on the
# real duck: nothing of the simulator, the physics, the page or the world, and no Duck Core.
BOARD_SIDE = ("duckmind.duckbody", "duckmind.eye", "duckmind.curiosity", "duckmind.percept")
NOT_ON_THE_BOARD = (
    "duckmind.world",
    "duckmind.worldport",  # phase 4: Duck Core's and the simulator's side of the boundary
    "duckmind.engines",
    "duckmind.simweb",
    "duckmind.livebody",
    "duckmind.bodyhost",
    "duckmind.simulator",
    "duckmind.sim_live",
    "duckmind.sim_pet",
    "duckmind.local_env",
    "mujoco",
    "mjlab_microduck",
)
DUCKCORE_PUBLIC = frozenset({"duckmind.duckcore.fixed", "duckmind.duckcore.rng"})
# Phase 4: the neutral world port, the engine revisions and LegacyWorld.
WORLDPORT = "duckmind.worldport"
ENGINES = "duckmind.engines"
LEGACY_WORLD = "duckmind.world.legacy"


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


@cache
def _modules() -> dict[str, Path]:
    """Every module under src/, by dotted name (a package by its own name)."""
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join(parts)] = path
    return found


def _read(path: Path, module: str) -> tuple[tuple[str, int], ...]:
    """(imported module, line) for every import in the file of ``module``. ``from P import N``
    counts as the submodule ``P.N`` when that is (or, for Duck Core's public modules, will be)
    a module, else as ``P`` itself; ``from duckmind import world`` counts as ``duckmind.world``
    even before that package exists."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    known = set(_modules()) | DUCKCORE_PUBLIC
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join([*parts[: len(parts) - node.level + 1], *filter(None, [base])])
            for alias in node.names:
                full = f"{base}.{alias.name}"
                found.append((full if full in known else base, node.lineno))
                if full not in known and any(
                    _under(full, prefix) and not _under(base, prefix) for prefix in NOT_FOR_THE_MIND
                ):
                    found.append((full, node.lineno))
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func, first = node.func, node.args[0].value
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"import_module", "__import__"} and isinstance(first, str):
                found.append((first, node.lineno))
    return tuple(found)


@cache
def _imports(module: str) -> tuple[tuple[str, int], ...]:
    return _read(_modules()[module], module)


def _parents(module: str) -> list[str]:
    """The packages whose ``__init__`` runs before ``module`` (``duckmind`` first)."""
    parts = module.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts))]


def _closure(roots: list[str]) -> dict[str, str | None]:
    """Every src module the roots load, with the module that first pulled it in."""
    seen: dict[str, str | None] = {root: None for root in roots}
    queue = list(roots)
    while queue:
        module = queue.pop(0)
        for name in [name for name, _line in _imports(module)] + _parents(module):
            for candidate in [*_parents(name), name]:
                if candidate in _modules() and candidate not in seen:
                    seen[candidate] = module
                    queue.append(candidate)
    return seen


def _chain(seen: dict[str, str | None], module: str) -> str:
    links = [module]
    while (parent := seen[links[-1]]) is not None:
        links.append(parent)
    return " ← ".join(links)


def test_duck_core_loads_nothing_of_the_world() -> None:
    roots = [name for name in _modules() if _under(name, DUCKCORE)]
    assert len(roots) > 10, "the duckmind.duckcore modules were not found"
    seen = _closure(roots)
    problems = [
        f"{_chain(seen, module)} imports {name} (line {line})"
        for module in sorted(seen)
        for name, line in _imports(module)
        if any(_under(name, prefix) for prefix in NOT_FOR_THE_MIND)
    ]
    assert not problems, "Duck Core must not load the world (spec R1):\n" + "\n".join(problems)


def test_the_world_and_the_eyes_use_no_duck_core_internals() -> None:
    modules = [name for name in _modules() if any(_under(name, side) for side in SENSE_SIDE)]
    problems = [
        f"{module} imports {name} (line {line})"
        for module in sorted(modules)
        for name, line in _imports(module)
        if _under(name, DUCKCORE) and name not in DUCKCORE_PUBLIC
    ]
    assert not problems, (
        "duckmind.world and duckmind.eye may use only "
        + ", ".join(sorted(DUCKCORE_PUBLIC))
        + " from Duck Core:\n"
        + "\n".join(problems)
    )


def test_the_board_side_loads_nothing_of_the_simulator() -> None:
    roots = [name for name in _modules() if any(_under(name, side) for side in BOARD_SIDE)]
    for side in BOARD_SIDE:
        assert any(_under(name, side) for name in roots), f"the {side} modules were not found"
    seen = _closure(roots)
    problems = [
        f"{_chain(seen, module)} imports {name} (line {line})"
        for module in sorted(seen)
        for name, line in _imports(module)
        if any(_under(name, prefix) for prefix in NOT_ON_THE_BOARD) or _under(name, DUCKCORE)
    ]
    assert not problems, (
        "duck-body, duck-eye, the curiosity Act Runner and percept-v1 run on the board: no "
        "simulator, physics, page or world code, and no Duck Core:\n" + "\n".join(problems)
    )


def _duckmind_imports(module: str) -> set[str]:
    return {name for name, _line in _imports(module) if _under(name, "duckmind")}


def test_the_world_port_is_neutral() -> None:
    """duckmind.worldport: the standard library and percept-v1 only (phase-4 design A4)."""
    assert WORLDPORT in _modules(), "duckmind.worldport was not found"
    assert _duckmind_imports(WORLDPORT) <= {"duckmind.percept"}, _duckmind_imports(WORLDPORT)
    # (every module loads the ``duckmind`` package's own __init__; nothing of a side comes along)
    sides = (DUCKCORE, ENGINES, "duckmind.world", "duckmind.simweb", *BOARD_SIDE[:3])
    seen = _closure([WORLDPORT])
    loaded = sorted(module for module in seen if any(_under(module, side) for side in sides))
    assert not loaded, [_chain(seen, module) for module in loaded]


def test_every_engine_revision_loads_nothing_of_the_world() -> None:
    """Rule R1 for the frozen v4 engine and the migration seam too (``duckmind.engines``)."""
    roots = [name for name in _modules() if _under(name, ENGINES)]
    assert f"{ENGINES}.v4_0.duckcore.engine" in roots and f"{ENGINES}.migrate" in roots
    seen = _closure(roots)
    problems = [
        f"{_chain(seen, module)} imports {name} (line {line})"
        for module in sorted(seen)
        for name, line in _imports(module)
        if any(_under(name, prefix) for prefix in NOT_FOR_THE_MIND)
    ]
    assert not problems, "an engine must not load the world (spec R1):\n" + "\n".join(problems)


def test_legacy_world_uses_the_port_and_duck_cores_public_maths_only() -> None:
    """LegacyWorld draws with the duck's own key (duckcore.rng, duckcore.fixed) and speaks
    through the port (worldport, percept-v1): no Duck Core internals, no engine."""
    assert LEGACY_WORLD in _modules(), "duckmind.world.legacy was not found"
    allowed = DUCKCORE_PUBLIC | {WORLDPORT, "duckmind.percept"}
    assert _duckmind_imports(LEGACY_WORLD) <= allowed, _duckmind_imports(LEGACY_WORLD) - allowed


def test_the_import_reader_sees_every_way_in(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        textwrap.dedent(
            """
            import importlib
            from typing import TYPE_CHECKING

            from duckmind import world
            from duckmind.duckcore import DuckCore
            from . import fixed
            from ..duckcore.engine import SOURCES

            if TYPE_CHECKING:
                import duckmind.eye.light

            def later():
                import mujoco

                page = importlib.import_module("duckmind.simweb.live")
                return page, __import__("mjlab_microduck")
            """
        ),
        encoding="utf-8",
    )
    found = {name for name, _line in _read(probe, "duckmind.duckcore.probe")}
    assert found >= {
        "duckmind.world",
        "duckmind.duckcore",
        "duckmind.duckcore.fixed",
        "duckmind.duckcore.engine",
        "duckmind.eye.light",
        "mujoco",
        "duckmind.simweb.live",
        "mjlab_microduck",
    }
    # …and on the real tree: nested imports, ``from package import module``, the closure.
    assert "duckmind.duckcore.engine" in dict(_imports("duckmind.duckcore.excursion"))
    assert "duckmind.duckcore.commands" in dict(_imports("duckmind.duckcore.api"))
    assert "duckmind.expression_sweep.clips" in _closure(["duckmind.duckcore.api"])
    assert _parents("duckmind.duckcore.api") == ["duckmind", "duckmind.duckcore"]
