"""The known home: the lasting map a duck that has lived in its home carries, as a fixture.

A fixture is ``data/known_home-<home>.json``: the map's saved state (``LastMap.to_state``) and ``origin``, the world
pose of the map's own frame (where the roam that made it began). :func:`install` gives it to a fresh brain standing
anywhere in that home: the map's odometry correction is set so that the map lies where it truly is — a duck that knows
its home and knows where it is in it, which is the "known home" every go-there measurement since 2026-11-06 used.

``real_home``'s committed fixture is that measurement's own map (a 40-minute roam of the simulated duck on the v8 walk),
kept so the numbers stay comparable. :func:`roam` makes one for any home, headless and seeded, so it can be made again.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = "bench-knownhome-v1"
DATA = Path(__file__).resolve().parent / "data"
NONE = "none"  # a duck that has never seen its home


def _compose(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def _inverse(a: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2])


def path_of(known: str | None, home: str) -> Path | None:
    if known == NONE:
        return None
    if known:
        p = Path(known)
        return p if p.suffix == ".json" else DATA / f"known_home-{known}.json"
    return DATA / f"known_home-{home}.json"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} fixture")
    return data  # type: ignore[no-any-return]


def install(brain: Any, known: str | None, truth: Sequence[float], home: str = "real_home") -> bool:
    """Give ``brain`` (standing at world pose ``truth``) the known home ``known`` (a name, a path, None: the home's own,
    ``"none"``: nothing). False when there is nothing to give."""
    path = path_of(known, home)
    if path is None:
        return False
    fixture = _load(str(path))
    lm = brain.lastmap
    lm.load_state(fixture["map"], t_now=lm._now_s())
    odom = brain._odom_now
    want = _compose(_inverse(fixture["origin"]), truth)
    th = want[2] - odom[2]
    c, s = math.cos(th), math.sin(th)
    lm.corr = (want[0] - (c * odom[0] - s * odom[1]), want[1] - (s * odom[0] + c * odom[1]), th)
    lm._dirty = True
    return True


def roam(
    home: str, seed: int, minutes: float, *, collision: str, policies: str, log: Any = None
) -> dict[str, Any]:
    """A fixture made now: a fresh duck (``seed``) left alone in ``home`` for ``minutes`` of its own time, headless."""
    from duckmind.bench import runner as RN

    RN.prepare_env()
    from duckmind.neuro.genome import Genome
    from duckmind.world import homelayout
    from duckmind.world import neuro_play_rig as R

    RN._live_body(collision, policies)
    layout = homelayout.load(home)
    spawn = homelayout.spawn_pose(layout, "nest")
    sj = spawn.as_json()
    x, y, yaw = sj["x_mm"] / 1000.0, sj["y_mm"] / 1000.0, sj["heading_mrad"] / 1000.0
    x, y = (
        x + 0.45 * math.cos(yaw),
        y + 0.45 * math.sin(yaw),
    )  # (standing in front of its dock, not on it)
    with R.PlayNeuro(
        Genome.from_seed(seed, gd=50), seed, x=x, y=y, yaw=yaw, home=home, when=RN.WHEN
    ) as rig:
        RN.Watch(rig, seed, "main", (x, y), False)
        RN._link_as_live(rig, collision, policies)
        rig.brain.can_pivot = True
        start = rig.pose()
        steps = int(minutes * 60.0 / 30.0)
        for n in range(steps):
            rig.run(30.0)
            if log is not None and (n + 1) % 10 == 0:
                log(
                    f"roam {0.5 * (n + 1):.0f} min, {len(rig.brain.lastmap.cells)} cells, at {tuple(round(v, 2) for v in rig.pose())}"
                )
        lm = rig.brain.lastmap
        mine = lm.pose(rig.brain._odom_now)
        origin = _compose(rig.pose(), _inverse(mine))
        return {
            "schema": SCHEMA,
            "home": home,
            "layout": str(layout.fingerprint),
            "made": {
                "how": "headless roam",
                "seed": int(seed),
                "minutes": float(minutes),
                "start": [round(v, 3) for v in start],
            },
            "origin": [round(v, 5) for v in origin],
            "map": lm.to_state(),
        }


__all__ = ["DATA", "NONE", "SCHEMA", "install", "path_of", "roam"]
