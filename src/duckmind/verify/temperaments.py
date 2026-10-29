"""The four named temperaments machine verification runs (MACHINE-VERIFY.md, "L1 / L2").

The axes are the ones of ``tests/test_neuro_brain.py::TEMPERAMENTS`` (copied here so that tools
need not import a test module; ``tests/test_machine_verify_temperaments.py`` checks the two stay
equal). A duck is one of these at growth day 120, with the brain seed varied per trial.
"""

from __future__ import annotations

import dataclasses

from duckmind.neuro.contract import BEHAVIOURS
from duckmind.neuro.genome import Genome

GROWTH_DAY = 120

TEMPERAMENTS: dict[str, tuple[dict[str, float], str]] = {
    "bold_explorer": (
        dict(
            boldness=0.9,
            activity=0.9,
            sociability=0.3,
            exploration=0.9,
            wit=0.7,
            chronotype=0.3,
            tempo=0.6,
        ),
        "explorer",
    ),
    "shy_snuggler": (
        dict(
            boldness=0.1,
            activity=0.3,
            sociability=0.9,
            exploration=0.2,
            wit=0.4,
            chronotype=0.5,
            tempo=0.4,
        ),
        "snuggler",
    ),
    "lazy_dreamer": (
        dict(
            boldness=0.5,
            activity=0.1,
            sociability=0.4,
            exploration=0.3,
            wit=0.3,
            chronotype=0.8,
            tempo=0.2,
        ),
        "dreamer",
    ),
    "social_songbird": (
        dict(
            boldness=0.6,
            activity=0.6,
            sociability=0.8,
            exploration=0.5,
            wit=0.8,
            chronotype=0.4,
            tempo=0.7,
        ),
        "songbird",
    ),
}
PRESET_NAMES: tuple[str, ...] = tuple(TEMPERAMENTS)
# "大膽型" (MACHINE-VERIFY.md M3): the temperament whose boldness axis is the highest one.
BOLD_PRESETS: frozenset[str] = frozenset({"bold_explorer"})


def genome(name: str, seed: int) -> Genome:
    """The named temperament at growth day 120, with ``seed`` as its brain/motor seed. Call
    ``duckmind.neuro.integration.install_default_skills()`` first: unlocks and gains are computed
    against the behaviours registered at this moment."""
    axes, branch = TEMPERAMENTS[name]
    base = Genome.from_axes(axes, primary=branch, branch=branch, gd=GROWTH_DAY)
    return dataclasses.replace(base, seed=int(seed))


def signature_acts(name: str) -> tuple[str, ...]:
    """The behaviours that are this temperament's branch signatures (contract.BEHAVIOURS)."""
    branch = TEMPERAMENTS[name][1]
    return tuple(sorted(k for k, spec in BEHAVIOURS.items() if spec.branch == branch))
