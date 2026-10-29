"""The life duck of the growth integration against the explorer it replaces
(docs/design/GROWTH-G1-2026-10-10.md): the same seed (4107, the bold explorer's nature) at growth
day 0 (what 蛋黃 is on the day it hatches) and at growth day 60 (what the live duck was before).

They are not among the four temperaments machine verification gates on (``temperaments.py`` and
the v1–v3 runners are frozen: ``tests/test_machine_verify_v4_outcomes.py`` checks their bytes), so
they are named here only, for comparison runs. ``install()`` makes ``temperaments.genome`` know
them; a comparison run puts a ``sitecustomize.py`` that calls it on ``PYTHONPATH`` (so spawned
workers have it too) and then names them: ``--presets danhuang_gd0,explorer_gd60``.
"""

from __future__ import annotations

import dataclasses

from duckmind.neuro.genome import Genome

LIFE_FIXTURES: dict[str, tuple[int, int]] = {"danhuang_gd0": (4107, 0), "explorer_gd60": (4107, 60)}


def genome(name: str, seed: int) -> Genome:
    """The named life fixture with ``seed`` as its brain/motor seed (call
    ``integration.install_default_skills()`` first, as for ``temperaments.genome``)."""
    life_seed, gd = LIFE_FIXTURES[name]
    return dataclasses.replace(Genome.from_seed(life_seed, gd=gd), seed=int(seed))


def install() -> None:
    """Make ``duckmind.verify.temperaments.genome`` answer for the life fixtures too."""
    from duckmind.verify import temperaments

    original = temperaments.genome

    def genome_with_life(name: str, seed: int) -> Genome:
        return genome(name, seed) if name in LIFE_FIXTURES else original(name, seed)

    temperaments.genome = genome_with_life  # type: ignore[assignment]
