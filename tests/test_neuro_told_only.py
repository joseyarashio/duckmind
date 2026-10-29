"""The tricks a person can teach (bow, lie_low, spin, flip) are channels with no wiring of their own: a
duck that is never told is the duck it was before they existed. Taking them out of the repertoire must
change nothing of what the duck does by itself — its noise, its choices, its holds — and the channels
themselves are never the duck's own pick."""

from __future__ import annotations

import dataclasses

import pytest
from test_neuro_brain import _genome

from duckmind.neuro import select as S

DT = 0.05
TRICK_CHANNELS = frozenset({"bow", "lie_low", "spin", "flip"})
STANDING = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)


def _features() -> list[float]:
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.6
    feat[S.F["curiosity"]] = 0.7
    feat[S.F["energy"]] = 0.9
    feat[S.F["boredom"]] = 0.3
    return feat


def _run(sel: S.Selector, seconds: float = 120.0) -> list[tuple[str, bool]]:
    feat = _features()
    return [sel.step(DT, feat, STANDING, False, 0.0) for _ in range(int(seconds / DT))]


@pytest.mark.parametrize("temperament", ["bold_explorer", "shy_snuggler", "lazy_dreamer", "social_songbird"])
@pytest.mark.parametrize("seed", [1, 4107, 90210])
def test_a_duck_never_told_chooses_the_same_with_or_without_the_tricks(
    temperament: str, seed: int
) -> None:
    genome = _genome(temperament)
    assert TRICK_CHANNELS <= genome.unlocked  # (the tricks are in this duck's repertoire)
    bare = dataclasses.replace(genome, unlocked=genome.unlocked - TRICK_CHANNELS)
    with_tricks = S.Selector(genome, seed)
    without = S.Selector(bare, seed)
    assert not TRICK_CHANNELS & set(without.names)
    got = _run(with_tricks)
    expected = _run(without)
    first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
    assert first is None, f"the choices part at tick {first}: {got[first]} against {expected[first]}"
    assert not TRICK_CHANNELS & {name for name, _ in got}


def test_a_told_channel_is_taken_up_only_when_a_claim_gives_it_the_body() -> None:
    sel = S.Selector(_genome("bold_explorer"), seed=3)
    _run(sel, 20.0)
    assert sel.names[sel.current] not in TRICK_CHANNELS
    sel.claim_next("bow")
    name, new = sel.step(DT, _features(), STANDING, False, 0.0)
    assert new and name == "bow"
