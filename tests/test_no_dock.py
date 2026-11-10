"""A body without a charging dock (``has_dock`` false: a real Microduck, until a dock exists).

The brain does not look for a dock or walk home to one; its hunger still follows the pack, and
falls when the family swaps it. Every simulated body keeps a dock (with a nest or without one):
its brain is the brain it was, tick for tick.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_neuro_dock import _frame
from test_real_duck_body import _body, duck  # noqa: F401 - a fake real duck behind its bridge

from duckmind.neuro import dock as D
from duckmind.neuro import select as S
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import preset

TICK_MS = 20
POSE = D.DockPose(1500.0, 100.0, math.pi - 0.1)  # its dock's tag in view (when there is one)


def low_pack_run(brain: NeuroBrain, seconds: float, *, tag: bool = False) -> list[tuple[Any, ...]]:
    """A standing duck whose pack reads 36 % and whose hunger is high: every tick's act."""
    brain.drives.hunger = 0.8
    out = []
    for k in range(round(seconds * 1000 / TICK_MS)):
        t_ms = TICK_MS * k
        cmd, tel = brain.step(t_ms, _frame(t_ms, pose=POSE if tag else None))
        mode = tel.dock["mode"] if tel.dock else None
        out.append((cmd.behaviour, cmd.docking, mode, round(cmd.vx, 4), round(cmd.vyaw, 4)))
    return out


@pytest.mark.parametrize("name,seed", [("snuggler", 3), ("explorer", 11), ("dreamer", 5)])
def test_without_a_dock_a_low_pack_never_sends_it_looking_for_one(name: str, seed: int) -> None:
    with_dock = NeuroBrain(preset(name), seed=seed)
    assert any(act == "return_dock" for act, *_ in low_pack_run(with_dock, 60.0))  # (as today)
    brain = NeuroBrain(preset(name), seed=seed)
    brain.set_has_dock(False)
    run = low_pack_run(brain, 60.0)
    assert not [row for row in run if row[0] == "return_dock"]
    assert {mode for _, _, mode, _, _ in run} <= {"idle", None}  # the docking program never ran
    assert not any(docking for _, docking, *_ in run)
    assert brain._feat[S.F["homing"]] == 0.0 and brain._feat[S.F["home_pull"]] == 0.0
    assert brain._feat[S.F["battery_low"]] > 0.5  # (the low pack still damps the walks away)


def test_without_a_dock_it_does_not_take_a_tag_like_sight_for_home() -> None:
    brain = NeuroBrain(preset("snuggler"), seed=3)
    brain.set_has_dock(False)
    run = low_pack_run(brain, 30.0, tag=True)
    assert not [row for row in run if row[0] == "return_dock" or row[1]]
    assert {mode for _, _, mode, _, _ in run} <= {"idle", None}


def test_without_a_dock_hunger_follows_the_pack_and_a_swapped_pack_ends_it() -> None:
    """robotd reports no charging flag (unknown reads as not charging): hunger is the pack's
    percentage alone. It rises while the pack is low and is gone a minute after a full pack is in."""
    brain = NeuroBrain(preset("snuggler"), seed=3)
    brain.set_has_dock(False)
    brain.drives.hunger = 0.0
    seated = {"posture": "sit"}
    for k in range(round(90_000 / TICK_MS)):
        brain.step(TICK_MS * k, _frame(TICK_MS * k, battery=0.34, **seated))
    hungry = brain.drives.hunger
    assert hungry == pytest.approx((0.58 - 0.34) / 0.3, abs=0.03)
    start = 90_000
    for k in range(round(90_000 / TICK_MS)):
        t_ms = start + TICK_MS * k
        frame = _frame(t_ms, battery=1.0, **seated)
        assert not frame.power.charging and not frame.power.on_dock
        brain.step(t_ms, frame)
    assert brain.drives.hunger < 0.02
    assert brain.behaviour != "return_dock"


def digest(run: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(repr(run).encode()).hexdigest()


@pytest.mark.parametrize("name,seed", [("snuggler", 3), ("explorer", 11), ("dreamer", 5)])
@pytest.mark.parametrize("tag", [True, False], ids=["nest-in-view", "no-nest-in-view"])
def test_a_body_with_a_dock_is_the_brain_it_was(name: str, seed: int, tag: bool) -> None:
    """The simulated duck, with a nest and without one, is told it has a dock: its acts are the
    same, tick for tick, as a brain that was told nothing (two minutes on a low pack: the walk
    home, the search, the docking program, or the search given up)."""
    untold = low_pack_run(NeuroBrain(preset(name), seed=seed), 120.0, tag=tag)
    told = NeuroBrain(preset(name), seed=seed)
    told.set_has_dock(True)
    assert digest(low_pack_run(told, 120.0, tag=tag)) == digest(untold)
    assert any(act == "return_dock" for act, *_ in untold)


# ── which body has a dock ────────────────────────────────────────────────────


class Brain:
    def __init__(self) -> None:
        self.told: list[bool] = []

    def set_has_dock(self, on: bool) -> None:
        self.told.append(on)


def test_the_driver_tells_the_brain_what_the_body_says() -> None:
    from duckmind.simweb.neuro_live import _sync_has_dock

    def told(body: Any) -> list[bool]:
        brain = Brain()
        _sync_has_dock(SimpleNamespace(brain=brain), body)
        return brain.told

    assert told(SimpleNamespace(has_dock=False)) == [False]
    assert told(SimpleNamespace(has_dock=True)) == [True]
    assert told(SimpleNamespace()) == [True]  # (a body that does not say: a simulated one)
    _sync_has_dock(
        SimpleNamespace(brain=object()), SimpleNamespace(has_dock=False)
    )  # (a stub lane)


def test_a_real_ducks_body_has_no_dock_until_its_bridge_says_it_has(
    tmp_path: Path,
    duck: Any,  # noqa: F811
) -> None:
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        assert body.real and body.has_dock is False
    finally:
        body.stop()
    duck.bridge.body_info["has_dock"] = True  # (the bridge started with --has-dock)
    body = _body(tmp_path, duck)
    try:
        body.start(timeout_s=8.0)
        assert body.real and body.has_dock is True
    finally:
        body.stop()


def test_every_simulated_body_has_a_dock_and_a_real_one_only_when_its_bridge_says_so() -> None:
    from duckmind.livebody import dock_of

    assert dock_of({"backend": "simulator"}) is True
    assert dock_of({"backend": "simulator", "has_dock": False}) is True  # (the world has its own)
    assert dock_of({}) is True  # (a body on this computer)
    assert dock_of({"backend": "real"}) is False
    assert dock_of({"backend": "real", "rehearsal": True}) is False
    assert dock_of({"backend": "real", "has_dock": True}) is True


def test_the_bridges_say_whether_the_duck_has_a_dock() -> None:
    from duckmind.remotebody import bridge as B
    from duckmind.remotebody import simbody

    args = B.bridge_parser().parse_args(["--token-file", "t"])
    assert args.has_dock is False
    assert B.bridge_parser().parse_args(["--token-file", "t", "--has-dock"]).has_dock is True
    body = SimpleNamespace(policy_hashes=lambda: None)
    assert simbody.body_info(body, as_real=True)["has_dock"] is False
    assert simbody.body_info(body, as_real=True, has_dock=True)["has_dock"] is True
