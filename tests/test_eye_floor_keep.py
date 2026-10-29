"""duck-eye keeps the floor it learnt (``--floor-model``): written after a learn that added samples."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from duckmind.eye import server as S


class _Model:
    def __init__(self) -> None:
        self.walks = 0

    def to_json(self) -> dict[str, Any]:
        return {"walks": self.walks, "samples": 10 * self.walks}


def test_the_learned_floor_is_written_after_each_useful_learn(tmp_path: Path) -> None:
    model = _Model()
    added = iter([12, 0, 7])

    def on_learn(event: Any) -> dict[str, Any]:
        model.walks += 1
        return {"floor": next(added)}

    core = SimpleNamespace(tasks={"floor": SimpleNamespace(model=model)}, on_learn=on_learn)
    path = tmp_path / "eye" / "floor.json"
    S._keep_floor(core, path)  # type: ignore[arg-type]
    assert core.on_learn("walk 1") == {"floor": 12}
    assert json.loads(path.read_text()) == {"walks": 1, "samples": 10}
    core.on_learn("walk 2 (nothing added: not written)")
    assert json.loads(path.read_text()) == {"walks": 1, "samples": 10}
    core.on_learn("walk 3")
    assert json.loads(path.read_text()) == {"walks": 3, "samples": 30}
    assert not path.with_suffix(".json.tmp").exists()


def test_no_floor_task_means_nothing_to_keep(tmp_path: Path) -> None:
    core = SimpleNamespace(tasks={}, on_learn=lambda event: {})
    S._keep_floor(core, tmp_path / "floor.json")  # type: ignore[arg-type]
    assert core.on_learn("x") == {}
    assert not (tmp_path / "floor.json").exists()
