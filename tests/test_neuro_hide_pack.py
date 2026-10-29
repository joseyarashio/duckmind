"""dock-home (2026-10-17): a duck whose pack runs low stops hiding and goes home; a real fright still hides.

The seed-1004 run of the charge-cycle control hid for a third of ten minutes with the pack at 39-41 % and
walked home too late: the hide channel's weights had no term for hunger or the low pack, so nothing in them
counted the pack while return_dock's grows with it. The weights are the selector's own table; these tests
run the real :class:`~duckmind.neuro.select.Selector` on a feature vector and read what wins.
"""

from __future__ import annotations

import pytest

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import select as S
from duckmind.neuro import skill_social_brain as SB
from duckmind.neuro.genome import preset

CTX = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)

# (hide_urge, fear, startled, looming): what is left of a fright
MILD = (0.25, 0.25, 0.0, 0.0)  # a leftover fear, the shelter is still the place
STRONG = (1.0, 0.9, 0.9, 0.5)  # a real fright, a threat close by

NAMES = ("snuggler", "explorer", "dreamer", "hatchling")


@pytest.fixture(scope="module")
def social() -> object:
    with SB.registered():
        yield SB


def _features(pack: float, fright: tuple[float, float, float, float], hiding: float) -> list[float]:
    urge, fear, startled, looming = fright
    f = S.empty_features()
    f[S.F["hide_urge"]] = urge
    f[S.F["fear"]] = fear
    f[S.F["startled"]] = startled
    f[S.F["looming"]] = looming
    f[S.F["hiding"]] = hiding
    f[S.F["shelter_known"]] = 0.8
    f[S.F["hunger"]] = max(0.0, min(1.0, (0.58 - pack) / 0.3))  # drives.py
    frame = C.SensorFrame(t_ms=0, power=C.PowerSense(battery=pack))
    f[S.F["battery_low"]] = B._battery_low(frame)  # the brain's own
    f[S.F["homing"]] = B._homing(frame)
    return f


def _wants(
    name: str,
    pack: float,
    fright: tuple[float, float, float, float],
    *,
    hiding: float = 1.0,
    going_home_used: float = 0.0,
) -> tuple[float, float]:
    """(hide, return_dock) as the competition scores them (with their satiation) for a duck in this state."""
    sel = S.Selector(preset(name), seed=1)
    sel.sat[sel.index("return_dock")] = going_home_used
    sel.step(0.05, _features(pack, fright, hiding), CTX, False, 0.0)
    scored = {n: sel._pre[i] * (1.0 - 0.85 * sel.sat[i]) for i, n in enumerate(sel.names)}
    return scored["hide"], scored["return_dock"]


@pytest.mark.parametrize("name", NAMES)
def test_a_low_pack_and_a_mild_leftover_fear_send_it_home(social: object, name: str) -> None:
    # the 1004 case: in its shelter, the pack at 40 %, only a little afraid. It hid on (its self-excitation
    # outscored going home); now going home wins, also when earlier tries have used 60 % of it up (with the
    # old weights, hide 1.32 against 1.0-1.2 for going home in all four presets)
    for used in (0.0, 0.6):
        hide, home = _wants(name, 0.40, MILD, going_home_used=used)
        assert home > hide, (name, used, hide, home)


@pytest.mark.parametrize("name", NAMES)
def test_a_real_fright_still_hides_with_the_pack_getting_low(social: object, name: str) -> None:
    # a threat close by and a shelter known: at 45 % (low, not empty) it hides; going home does not beat it
    hide, home = _wants(name, 0.45, STRONG)
    assert hide > home, (name, hide, home)
    # and the fright's pull is still there under the weights: starting a hide (not yet in) still beats going home
    hide, home = _wants(name, 0.45, STRONG, hiding=0.0)
    assert hide > home, (name, hide, home)


@pytest.mark.parametrize("name", NAMES)
def test_a_rested_duck_hides_as_it_did(social: object, name: str) -> None:
    # hunger is 0 from 58 % and battery_low from 46 %: the new terms are 0 there and nothing changed
    f = _features(0.60, MILD, 1.0)
    assert f[S.F["hunger"]] == 0.0 and f[S.F["battery_low"]] == 0.0
    hide, _ = _wants(name, 0.60, MILD)
    base = S.Selector(preset(name), seed=1)
    base.step(0.05, f, CTX, False, 0.0)
    w = S.W["hide"]
    by_hand = sum(wt * f[S.F[k]] for k, wt in w.items() if k not in ("hunger", "battery_low"))
    assert base._pre[base.index("hide")] == pytest.approx(base._gain[base.index("hide")] * by_hand)
    assert hide > 0.0


def test_the_weights_are_the_two_terms_asked_for(social: object) -> None:
    w = S.W["hide"]
    assert w["hunger"] < 0.0 and w["battery_low"] < 0.0
    assert w["hide_urge"] == 1.7  # the fright's own pull is untouched
