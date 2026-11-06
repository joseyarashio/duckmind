"""How the loaded walk policy answers a twist, and the twist that makes it answer as the brain's walk did.

Every steering number of the brain — the heading gains, the arcs that turn away from things, the drift trim, the turn
budgets of the reflexes, proprio's own-turn model, the docking feed-forward — was tuned on Pollen's walk policy v5, which
turns about half as much as it is told to while walking (0.55 of the command, and a little to the right when told to go
straight) and walks at about 0.4 of the commanded speed. Policy v8 (2026-11-05) turns 1.1-1.3 times the command: the same
brain on it turned twice as hard as it meant to and ended its errands stuck (go-there 0/10).

So the link asks the loaded walk for the twist that gives the response the brain was tuned with. A measured table per walk
file (keyed by the walk file's sha256, ``walk_response.json`` next to this module) holds the path speed and the turning rate
for a grid of commands (vx 0.30-0.55 m/s, vyaw -1..1 rad/s), measured on the live robotd in the simulator with its servo
command delay. One entry is the reference (v5: the policy the brain was tuned on); for any other walk with a table the link
sends the command whose measured answer on that walk equals the reference's measured answer to the brain's command (the
path speed and the turning rate together, so an arc keeps its radius). The reference walk, or a walk without a table, gets
the command unchanged. Other forward commands (slower than 0.30) and the docking reverse are never changed.

Curiosity's back arc (the other reversing twist: (−0.40, ±1.0) for ``BACK_ARC_MS`` = 1 s, a reflex's escape alike) gets
its turn mapped from a second table per walk (``back_arc``): the turn (rad/s) and the distance moved (m/s) over the first
second of a 1 s reverse arc, measured on the live robotd (2026-11-06, from standing and from walking at 0.35). In that
second v5 turns ≈ 24° left / 38° right and v8 sent the same twist ≈ 60° either way. proprio judges a graze by the heading
swing its v5 own-turn model does not explain (``brushed``), and v8 backs off fast enough to count as progress, so a v8 back
arc that lasted past proprio's 1 s spin-up was read as a graze (live wall check: 15 of v8's 63 brushes, 0 of v5's 48).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).with_name("walk_response.json")
WALK_FILE = "velstand.onnx"
FIX_ROUNDS = (
    4  # alternate turning-rate and speed solves this many times (the speed barely moves the rate)
)


@dataclass(frozen=True)
class WalkResponse:
    """One walk policy's measured answer: ``speed[i][j]`` (m/s along the path) and ``yaw[i][j]`` (rad/s, + = left) to
    the command (``vxs[i]``, ``vyaws[j]``), on the ``body`` collision model with a ``delay_ms`` servo command delay."""

    walk_sha256: str
    name: str
    body: str
    delay_ms: float
    vxs: tuple[float, ...]
    vyaws: tuple[float, ...]
    speed: tuple[tuple[float, ...], ...]
    yaw: tuple[tuple[float, ...], ...]
    # curiosity's 1 s back arc (vx −0.40…−0.30): turn and path speed over its first second (None: not measured)
    back_arc: WalkResponse | None = None

    @staticmethod
    def of(entry: Mapping[str, object]) -> WalkResponse:
        def grid(key: str) -> tuple[tuple[float, ...], ...]:
            rows = entry[key]
            assert isinstance(rows, Sequence)
            return tuple(tuple(float(v) for v in row) for row in rows)

        vxs = tuple(float(v) for v in entry["vx"])  # type: ignore[union-attr]
        vyaws = tuple(float(v) for v in entry["vyaw"])  # type: ignore[union-attr]
        back = entry.get("back_arc")
        out = WalkResponse(
            walk_sha256=str(entry["walk_sha256"]),
            name=str(entry["name"]),
            body=str(entry["body"]),
            delay_ms=float(entry["delay_ms"]),  # type: ignore[arg-type]
            vxs=vxs,
            vyaws=vyaws,
            speed=grid("speed"),
            yaw=grid("yaw_rate"),
            back_arc=None
            if back is None
            else WalkResponse.of(
                {**{k: entry[k] for k in ("walk_sha256", "name", "body", "delay_ms")}, **back}  # type: ignore[dict-item]
            ),
        )
        if list(vxs) != sorted(set(vxs)) or list(vyaws) != sorted(set(vyaws)):
            raise ValueError(f"walk response {out.name}: the grid is not increasing")
        for table in (out.speed, out.yaw):
            if len(table) != len(vxs) or any(len(row) != len(vyaws) for row in table):
                raise ValueError(
                    f"walk response {out.name}: the table is not {len(vxs)}×{len(vyaws)}"
                )
        return out

    def covers(self, vx: float) -> bool:
        return self.vxs[0] <= vx <= self.vxs[-1]

    def _row_at(self, table: tuple[tuple[float, ...], ...], vx: float) -> list[float]:
        """``table`` along vyaw at forward command ``vx`` (linear between the grid's rows, held at its ends)."""
        i, f = _locate(self.vxs, vx)
        return [a + f * (b - a) for a, b in zip(table[i], table[i + 1], strict=True)]

    def _column_at(self, table: tuple[tuple[float, ...], ...], vyaw: float) -> list[float]:
        """``table`` along vx at turn command ``vyaw``."""
        j, f = _locate(self.vyaws, vyaw)
        return [row[j] + f * (row[j + 1] - row[j]) for row in table]

    def at(self, vx: float, vyaw: float) -> tuple[float, float]:
        """(path speed, turning rate) for the command (``vx``, ``vyaw``), bilinear in the grid, held at its edges."""
        out = []
        for table in (self.speed, self.yaw):
            row = self._row_at(table, vx)
            j, f = _locate(self.vyaws, vyaw)
            out.append(row[j] + f * (row[j + 1] - row[j]))
        return out[0], out[1]

    def vyaw_for(self, vx: float, rate: float, near: float) -> float:
        """The turn command that turns at ``rate`` at forward command ``vx`` (the one nearest ``near`` when several do;
        the nearest reachable when none does)."""
        return _solve(self.vyaws, self._row_at(self.yaw, vx), rate, near)

    def vx_for(self, vyaw: float, speed: float, near: float) -> float:
        """The forward command that walks at path speed ``speed`` with turn command ``vyaw``."""
        return _solve(self.vxs, self._column_at(self.speed, vyaw), speed, near)


def _locate(grid: tuple[float, ...], x: float) -> tuple[int, float]:
    """The grid segment holding ``x`` and the fraction along it (clamped to the grid)."""
    if x <= grid[0]:
        return 0, 0.0
    if x >= grid[-1]:
        return len(grid) - 2, 1.0
    for i in range(len(grid) - 1):
        if grid[i] <= x <= grid[i + 1]:
            return i, (x - grid[i]) / (grid[i + 1] - grid[i])
    return len(grid) - 2, 1.0


def _solve(grid: tuple[float, ...], values: list[float], target: float, near: float) -> float:
    """``x`` on ``grid`` where the piecewise-linear ``values`` equal ``target``: of every crossing the one nearest
    ``near``; with no crossing, the grid point whose value comes nearest (then the one nearest ``near``)."""
    best: tuple[float, float] | None = None
    for i in range(len(grid) - 1):
        a, b = values[i], values[i + 1]
        lo, hi = min(a, b), max(a, b)
        if not lo <= target <= hi:
            continue
        x = grid[i] if a == b else grid[i] + (target - a) / (b - a) * (grid[i + 1] - grid[i])
        if best is None or abs(x - near) < best[0]:
            best = (abs(x - near), x)
    if best is not None:
        return best[1]
    return min(
        zip(grid, values, strict=True), key=lambda p: (abs(p[1] - target), abs(p[0] - near))
    )[0]


@dataclass(frozen=True)
class WalkMap:
    """From the brain's twist (a command for the ``reference`` walk) to the command for the ``loaded`` walk, and back."""

    reference: WalkResponse
    loaded: WalkResponse

    def command(self, vx: float, vyaw: float) -> tuple[float, float]:
        """The twist for the loaded walk that answers as the reference walk answers (``vx``, ``vyaw``)."""
        if vx <= 0.0 or not self.reference.covers(vx):
            return vx, vyaw
        return _matched(self.reference, self.loaded, vx, vyaw)

    def planned(self, vx: float, vyaw: float) -> tuple[float, float]:
        """The brain's twist that :meth:`command` turns into (``vx``, ``vyaw``) (what the brain meant by what the body
        was told: its senses read the body's twist in the units it plans in)."""
        if vx <= 0.0 or not self.loaded.covers(vx):
            return vx, vyaw
        return _matched(self.loaded, self.reference, vx, vyaw)

    def back_arc(self, vx: float, vyaw: float) -> tuple[float, float]:
        """The reverse twist for the loaded walk whose 1 s back arc turns as far as the reference walk's back arc
        (``vx``, ``vyaw``) does, at the same ``vx`` (the distance backed off is close already: at −0.40 v5 0.09-0.10 m,
        v8 0.08 m, and a slower reverse backs off less on both); unchanged when either walk's back arc was not measured
        or ``vx`` is outside it."""
        ref, loaded = self.reference.back_arc, self.loaded.back_arc
        if (
            vx >= 0.0
            or ref is None
            or loaded is None
            or not ref.covers(vx)
            or not loaded.covers(vx)
        ):
            return vx, vyaw
        return vx, round(loaded.vyaw_for(vx, ref.at(vx, vyaw)[1], vyaw), 4)

    def planned_back_arc(self, vx: float, vyaw: float) -> tuple[float, float]:
        """The brain's back-arc twist that :meth:`back_arc` turns into (``vx``, ``vyaw``)."""
        ref, loaded = self.reference.back_arc, self.loaded.back_arc
        if (
            vx >= 0.0
            or ref is None
            or loaded is None
            or not ref.covers(vx)
            or not loaded.covers(vx)
        ):
            return vx, vyaw
        return vx, round(ref.vyaw_for(vx, loaded.at(vx, vyaw)[1], vyaw), 4)


def _matched(
    source: WalkResponse, target: WalkResponse, vx: float, vyaw: float
) -> tuple[float, float]:
    speed, rate = source.at(vx, vyaw)
    x, w = vx, vyaw
    for _ in range(FIX_ROUNDS):
        w = target.vyaw_for(x, rate, w)
        x = target.vx_for(w, speed, x)
    return round(x, 4), round(w, 4)


@dataclass(frozen=True)
class WalkTable:
    """Every measured walk (by sha256) and which one is the reference."""

    reference: str
    walks: Mapping[str, WalkResponse]

    def map_for(self, walk_sha256: str | None, body: str | None, delay_ms: float) -> WalkMap | None:
        """The map for a body running the walk ``walk_sha256`` on the collision model ``body`` with a ``delay_ms`` servo
        command delay; None (send the brain's twist as it is) for the reference walk, an unmeasured walk, or a walk
        measured on another body or with another delay."""
        if walk_sha256 is None or walk_sha256 == self.reference:
            return None
        loaded, ref = self.walks.get(walk_sha256), self.walks.get(self.reference)
        if loaded is None or ref is None or (body is not None and loaded.body != body):
            return None
        if abs(loaded.delay_ms - delay_ms) > 1e-6 or abs(ref.delay_ms - delay_ms) > 1e-6:
            return None
        return WalkMap(ref, loaded)


def load(path: Path = DATA) -> WalkTable:
    data = json.loads(path.read_text(encoding="utf-8"))
    walks = {w.walk_sha256: w for w in (WalkResponse.of(e) for e in data["walks"])}
    if data["reference"] not in walks:
        raise ValueError("walk response table: the reference walk has no entry")
    return WalkTable(reference=str(data["reference"]), walks=walks)


def walk_sha256(policies: Path | None) -> str | None:
    """sha256 of the walk file in the policy folder ``policies`` (None when there is none)."""
    if policies is None:
        return None
    try:
        return hashlib.sha256((Path(policies) / WALK_FILE).read_bytes()).hexdigest()
    except OSError:
        return None


__all__ = ["DATA", "WalkMap", "WalkResponse", "WalkTable", "load", "walk_sha256"]
