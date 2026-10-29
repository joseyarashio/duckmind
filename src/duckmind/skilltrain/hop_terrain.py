"""The hop-over task's terrain: one strip across every tile (mjlab's terrain generator).

hop:feasibility (2026-09-28). A curriculum grid: rows = strip height (``hop_contract.HEIGHT_RANGE``,
0.5 → 4 cm, one height per row, so a row index *is* the height), columns = edge shape (``sharp``:
a rectangular strip; ``round``: the same strip with both long top edges rounded at
r = min(height, depth/2) — the Mac probe's two shapes). Each tile is a 2 m × 2 m floor box with the
strip across its full width, the near face at x = 1 m; the tile's spawn origin sits on the near face
(the reset event puts the duck 6–18 cm before it). Terrain geoms live in mjlab's ``terrain`` body,
so every contact sensor that matches ``terrain`` (feet, servo housings, head, trunk) sees the strip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

from duckmind.skilltrain import hop_contract as H

FLOOR_COLOR = (0.55, 0.55, 0.55, 1.0)
STRIP_COLOR = (0.62, 0.45, 0.30, 1.0)


def strip_geoms(body: mujoco.MjsBody, x0: float, y_mid: float, height: float, depth: float,
                width: float, edge: str) -> list[mujoco.MjsGeom]:  # fmt: skip
    """The strip's geoms (tile frame): the Mac probe's shapes (``hop_probe.Obstacle.geoms``)."""
    out = []
    h, d, w = height, depth, width / 2.0
    if edge == "sharp":
        out.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(d / 2, w, h / 2),
                                 pos=(x0 + d / 2, y_mid, h / 2)))  # fmt: skip
        return out
    r = H.round_radius(h, d)
    core = d - 2 * r
    if core > 1e-6:
        out.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(core / 2, w, h / 2),
                                 pos=(x0 + d / 2, y_mid, h / 2)))  # fmt: skip
    quat = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    centres = [x0 + r] if abs(core) <= 1e-6 else [x0 + r, x0 + d - r]
    for cx in centres:
        out.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(r, w, 0.0),
                                 pos=(cx, y_mid, h - r), quat=quat))  # fmt: skip
    low = h - r
    if low > 1e-6:
        for cx in (x0 + r / 2, x0 + d - r / 2):
            out.append(body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(r / 2, w, low / 2),
                                     pos=(cx, y_mid, low / 2)))  # fmt: skip
    return out


@dataclass(kw_only=True)
class StripTerrainCfg(SubTerrainCfg):
    """A floor tile with one strip across it; the height from the curriculum row."""

    edge: str = "sharp"
    depth: float = H.DEPTH_M
    rows: int = H.TERRAIN_ROWS
    face_x: float = H.FACE_X_IN_TILE

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        sx, sy = self.size
        row = min(self.rows - 1, max(0, int(math.floor(difficulty * self.rows + 1e-9))))
        height = H.height_of_row(row, self.rows)
        floor = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(sx / 2, sy / 2, 0.05),
                              pos=(sx / 2, sy / 2, -0.05))  # fmt: skip
        geoms = [TerrainGeometry(geom=floor, color=FLOOR_COLOR)]
        for g in strip_geoms(body, self.face_x, sy / 2, height, self.depth, sy, self.edge):
            geoms.append(TerrainGeometry(geom=g, color=STRIP_COLOR))
        return TerrainOutput(origin=np.array([self.face_x, sy / 2, 0.0]), geometries=geoms)


def terrain_cfg(rows: int = H.TERRAIN_ROWS, seed: int = 7) -> TerrainGeneratorCfg:
    """Curriculum mode: one column per edge shape, ``rows`` heights."""
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=True,
        size=H.TILE_M,
        border_width=1.0,
        border_height=0.1,  # a flat border: top at the floor (make_border centres it at −h/2)
        num_rows=rows,
        num_cols=len(H.EDGES),
        color_scheme="height",
        sub_terrains={e: StripTerrainCfg(edge=e, rows=rows, proportion=1.0) for e in H.EDGES},
        difficulty_range=(0.0, 1.0),
        add_lights=False,
    )


def height_of(levels, rows: int = H.TERRAIN_ROWS):
    """Strip height (m) of terrain rows (a torch tensor or an int)."""
    lo, hi = H.HEIGHT_RANGE
    return lo + (hi - lo) * (levels / max(1, rows - 1))


__all__ = ["StripTerrainCfg", "height_of", "strip_geoms", "terrain_cfg"]
