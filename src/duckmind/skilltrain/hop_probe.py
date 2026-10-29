"""Hop feasibility, part (a): what the v5 walk does today at a threshold or a low box edge.

hop:feasibility (2026-09-28, owner 「我希望還有加入小跳躍，遇到不高的障礙或是門檻 個性有一定程度勇敢的
會主動嘗試跳躍」). Before any training: which heights of thresholds the shipped v5 walk network
(velstand, the walk slot) already steps over, trips on or falls at — on both servo models, with
the per-duck domain randomisation of the Stage-1 gates, n ≥ 30 per cell.

**The scene.** The official ``scene_allcollisions.xml`` (the whole-body collision model velstand
was trained on, the owner's live simulator's) with one strip across the floor, 2 m wide and
``depth`` deep (4 cm: a door threshold, a book on its side), ``height`` 0.5–5 cm, its near face at
x = :data:`FACE_X`. Two edge shapes: ``sharp`` (a rectangular strip) and ``round`` (the same strip
with both long top edges rounded at r = min(height, depth/2): up to 2 cm a quarter circle from the
floor up — a bull-nosed saddle — above that a vertical face and a half-round top). The scene is a
folder of symlinks to the official files plus our scene file (:func:`scene_dir`); no official file
is copied or edited.

**A trial** (:func:`run_trial`): a fresh :class:`~duckmind.skilltrain.skillduck.SkillDuck` (one DR
draw, the same seed on both servo models) boots seated and rises through sitstand as the owner's
simulator does; it is then set down at the start pose (an evaluator's scene set-up, like the hand:
:meth:`~duckmind.world.headless_duck.HeadlessDuck.teleport`) 0.30–0.45 m before the face, heading
at it ±10°, stands 0.5–1.0 s, and walks at ``vx`` (0.30 / 0.40 / 0.55) with a heading hold on its
IMU yaw (vyaw = −3·error, ±1 rad/s, the arc-turn envelope: the brain walking at a thing it has seen). After the first
touch it keeps walking for up to 4 s ("tries to get over", phase 1's protocol with 1 s more), stops
1.0 s after it got over, or at once when robotd's fall rule fires; then 1.5 s (3 s after a fall)
with a zero twist.

**Outcomes** (first match): ``fall`` (robotd's rule: tilt > 60° for 0.2 s, from 0.5 s before the
touch), ``stuck`` (tilt > 40° for ≥ 1 s without the verdict: the nose-down blind spot), ``crossed``
(both feet and the trunk past the far face, upright), ``blocked`` (none of these by the end: stepping
in place against it, sliding along, or standing on it). Across outcomes: ``stumble`` (tilt ≥ 20°
after the touch — flat walking stays under 10°), ``toe_hits`` (foot contacts whose normal is more
horizontal than vertical: a toe against the face), servo-housing impacts (a housing > 2 N while
tilt < 45°), leg torque (peak actuator force per tick; p95, max, share > 0.4 N·m).

Run (resumable, one process per servo model; guard on the owner's :8765 realtime)::

    python -m duckmind.skilltrain.hop_probe run --servo mjcf --n 30 --out DIR
    python -m duckmind.skilltrain.hop_probe summarize --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCENE = "scene_allcollisions.xml"
FACE_X = 1.0  # world x of the obstacle's near face
WIDTH_M = 2.0
DEPTH_M = 0.04
HEIGHTS_M = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
EDGES = ("sharp", "round")
SPEEDS = (0.30, 0.40, 0.55)
START_D = (0.30, 0.45)  # trunk centre to the near face at the start (m)
AIM_DEG = 10.0  # heading at the face: normal ± this
HOLD_K, HOLD_MAX = 3.0, 1.0  # heading hold: vyaw = clip(−k·err, ±max); flat p5–p95 error ±10–17°
AFTER_TOUCH_S = 4.0
AFTER_CROSS_S = 1.0
WATCH_S, WATCH_FALL_S = 1.5, 3.0
NO_TOUCH_S = 10.0
STUMBLE_DEG = 20.0
STUCK_DEG, STUCK_S = 40.0, 1.0
HOUSING_N = 2.0
TORQUE_NM = 0.4
PAST_FOOT_M, PAST_TRUNK_M = (
    0.028,
    0.03,
)  # a sole is 5.4 cm long, its centroid 2.7 cm ahead of the heel
CONTROL_HZ = 50


# ── the obstacle (pure: no MuJoCo) ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Obstacle:
    """A strip across the floor; ``face_x``: its near face (the obstacle frame's origin)."""

    height_m: float
    edge: str = "sharp"
    depth_m: float = DEPTH_M
    width_m: float = WIDTH_M
    face_x: float = FACE_X

    def __post_init__(self) -> None:
        if self.edge not in EDGES:
            raise ValueError(f"edge {self.edge!r}: one of {EDGES}")
        if not (0.0 < self.height_m <= 0.20 and 0.0 < self.depth_m <= 1.0):
            raise ValueError("height 0–20 cm, depth 0–1 m")

    @property
    def radius_m(self) -> float:
        return 0.0 if self.edge == "sharp" else min(self.height_m, self.depth_m / 2.0)

    @property
    def name(self) -> str:
        return f"{self.edge}-{round(self.height_m * 1000):d}mm"

    def geoms(self) -> list[dict[str, Any]]:
        """MJCF geoms (type, size, pos, optional quat) in the world frame; the union is the
        strip's cross-section extruded along y."""
        h, d, w, x0 = self.height_m, self.depth_m, self.width_m / 2.0, self.face_x
        if self.edge == "sharp":
            return [{"type": "box", "size": (d / 2, w, h / 2), "pos": (x0 + d / 2, 0.0, h / 2)}]
        r = self.radius_m
        out: list[dict[str, Any]] = []
        core = d - 2 * r
        if core > 1e-6:
            out.append(
                {"type": "box", "size": (core / 2, w, h / 2), "pos": (x0 + d / 2, 0.0, h / 2)}
            )
        # cylinders along y (MuJoCo's cylinder axis is local z: turned 90° about x)
        quat = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
        centres = [x0 + r] if abs(core) <= 1e-6 else [x0 + r, x0 + d - r]
        for cx in centres:
            out.append({"type": "cylinder", "size": (r, w), "pos": (cx, 0.0, h - r), "quat": quat})
        low = h - r
        if low > 1e-6:
            for cx in (x0 + r / 2, x0 + d - r / 2):
                out.append({"type": "box", "size": (r / 2, w, low / 2), "pos": (cx, 0.0, low / 2)})
        return out

    def top_z(self, x_local: float) -> float:
        """The strip's surface height at ``x_local`` (0 = the near face; outside → 0)."""
        h, d, r = self.height_m, self.depth_m, self.radius_m
        if x_local < 0.0 or x_local > d:
            return 0.0
        if r <= 0.0:
            return h
        for cx in (r, d - r):
            dx = abs(x_local - cx)
            if (x_local < r and cx == r) or (x_local > d - r and cx == d - r):
                return max(0.0, h - r + math.sqrt(max(0.0, r * r - dx * dx)))
        return h

    def xml(self) -> str:
        rows = []
        for i, g in enumerate(self.geoms()):
            size = " ".join(f"{v:.6g}" for v in g["size"])
            pos = " ".join(f"{v:.6g}" for v in g["pos"])
            quat = f' quat="{" ".join(f"{v:.9g}" for v in g["quat"])}"' if "quat" in g else ""
            rows.append(
                f'        <geom name="hop_obst_{i}" type="{g["type"]}" size="{size}" '
                f'pos="{pos}"{quat} rgba="0.62 0.45 0.30 1"/>'
            )
        return "\n".join(rows)


def scene_dir(obstacle: Obstacle | None, root: str | os.PathLike[str], official: Path) -> Path:
    """A folder of symlinks to every official file with our ``scene_allcollisions.xml`` (the
    official scene text with the strip's geoms added to the worldbody). ``None``: flat."""
    name = obstacle.name if obstacle is not None else "flat"
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    for entry in official.iterdir():
        if entry.name == SCENE:
            continue
        link = folder / entry.name
        if not link.exists() and not link.is_symlink():
            try:
                link.symlink_to(entry)
            except FileExistsError:  # the other servo model's process made it first
                pass
    text = (official / SCENE).read_text()
    if obstacle is not None:
        marker = "</worldbody>"
        if marker not in text:
            raise ValueError("the official scene has no </worldbody>")
        text = text.replace(marker, obstacle.xml() + "\n    " + marker, 1)
    target = folder / SCENE
    if not target.exists() or target.read_text() != text:
        tmp = folder / f".{SCENE}.{os.getpid()}.tmp"
        tmp.write_text(text)
        tmp.replace(target)  # atomic: a concurrent reader sees the old or the new file, never half
    return folder


# ── outcomes from a recorded trial (pure numpy) ─────────────────────────────────────────────


def classify(rec: dict[str, Any], obstacle: Obstacle | None) -> dict[str, Any]:
    """The trial's outcome from its per-tick record (``t``, ``tilt``, ``fallen``, ``trunk_x``
    (obstacle frame), ``foot_x`` (N×2), ``toe_hits`` (N), ``housing`` (N: peak housing force),
    ``tau_legs`` (N×10), ``t_touch``, ``t_walk_end``)."""
    import numpy as np

    t = np.asarray(rec["t"], float)
    tilt = np.asarray(rec["tilt"], float)
    fallen = np.asarray(rec["fallen"], float) > 0.5
    trunk_x = np.asarray(rec["trunk_x"], float)
    foot_x = np.asarray(rec["foot_x"], float).reshape(-1, 2)
    t_touch = rec.get("t_touch")
    t0 = float(t_touch) if t_touch is not None else float(rec.get("t_walk", t[0]))
    after = t >= t0 - 0.5
    far = obstacle.depth_m if obstacle is not None else 0.0
    past = (
        (foot_x.min(axis=1) > far + PAST_FOOT_M)
        & (trunk_x > far + PAST_TRUNK_M)
        & (tilt < 45.0)
        & ~fallen
    )
    fell = bool((fallen & after).any())
    t_fall = float(t[np.flatnonzero(fallen & after)[0]]) if fell else None
    # stuck: tilt > 40° for ≥ 1 s without the verdict
    run, stuck = 0, False
    for i in np.flatnonzero(after):
        run = run + 1 if (tilt[i] > STUCK_DEG and not fallen[i]) else 0
        if run >= int(STUCK_S * CONTROL_HZ):
            stuck = True
            break
    crossed_idx = np.flatnonzero(past & after)
    crossed = bool(crossed_idx.size)
    t_cross = float(t[crossed_idx[0]]) if crossed else None
    # a fall anywhere after the touch is a fall (also on the far side: the step down)
    fell_after_cross = bool(fell and crossed and t_fall is not None and t_cross is not None
                            and t_cross < t_fall)  # fmt: skip
    if fell:
        outcome = "fall"
    elif stuck and not crossed:
        outcome = "stuck"
    elif crossed:
        outcome = "crossed"
    else:
        outcome = "blocked"
    window = after & (t <= float(rec.get("t_walk_end", t[-1])) + 0.5)
    max_tilt = float(tilt[window].max()) if window.any() else float("nan")
    tau = np.asarray(rec["tau_legs"], float).reshape(len(t), -1)
    tw = tau[window] if window.any() else tau
    housing = np.asarray(rec["housing"], float)
    impacts = int(((housing > HOUSING_N) & (tilt < 45.0) & window).sum())
    toe = np.asarray(rec["toe_hits"], float)
    return {
        "outcome": outcome,
        "fell_after_cross": fell_after_cross,
        "stumble": bool(max_tilt >= STUMBLE_DEG),
        "max_tilt_deg": round(max_tilt, 2),
        "t_fall_after_touch": None if t_fall is None else round(t_fall - t0, 3),
        "t_cross_after_touch": None if t_cross is None else round(t_cross - t0, 3),
        "toe_hits": int(toe[window].sum()),
        "housing_impact_ticks": impacts,
        "leg_tau_p95": round(float(np.percentile(np.abs(tw), 95)), 4) if tw.size else None,
        "leg_tau_max": round(float(np.abs(tw).max()), 4) if tw.size else None,
        "leg_tau_over_04_share": round(float((np.abs(tw).max(axis=1) > TORQUE_NM).mean()), 4)
        if tw.size
        else None,
    }


# ── one trial (MuJoCo, the Stage-1 gate duck) ───────────────────────────────────────────────


def seed_of(key: str, rep: int) -> int:
    return zlib.crc32(f"hop-a/{key}/{rep}".encode()) & 0x7FFFFFFF


def _yaw(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _make_duck_class():
    from duckmind.skilltrain.skillduck import SkillDuck

    class ProbeDuck(SkillDuck):
        """The gate duck, also counting foot contacts on the strip whose normal is more
        horizontal than vertical (a toe against the face), per physics substep."""

        def setup_probe(self, obstacle_geoms: Sequence[int]) -> None:
            import numpy as np

            self._obst = frozenset(int(g) for g in obstacle_geoms)
            self._feet_set = frozenset(int(g) for g in self.foot_geoms)
            self._toe_prev = np.zeros(2, dtype=bool)
            self.toe_tick = 0
            self.touch_tick = False
            self.robot_touch_any = False

        def _contacts(self, mujoco: Any):  # type: ignore[override]
            import numpy as np

            out = super()._contacts(mujoco)
            obst = getattr(self, "_obst", None)
            if not obst:
                return out
            d = self.data
            toe_now = np.zeros(2, dtype=bool)
            for i in range(int(d.ncon)):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in obst:
                    mine = g2
                elif g2 in obst:
                    mine = g1
                else:
                    continue
                if mine not in self.robot_geoms:
                    continue
                self.touch_tick = True
                self.robot_touch_any = True
                if mine in self._feet_set:
                    n = np.asarray(c.frame[:3], float)
                    if abs(n[2]) < math.hypot(n[0], n[1]):
                        toe_now[0 if mine == int(self.foot_geoms[0]) else 1] = True
            self.toe_tick += int((toe_now & ~self._toe_prev).sum())
            self._toe_prev = toe_now
            return out

    return ProbeDuck


_DUCK_CLASS: Any = None


@dataclass(frozen=True)
class Cell:
    height_m: float  # 0 = flat (the control)
    edge: str
    speed: float

    @property
    def key(self) -> str:
        return f"{self.edge}/{round(self.height_m * 1000):d}mm/v{self.speed:.2f}"

    def obstacle(self) -> Obstacle | None:
        return None if self.height_m <= 0 else Obstacle(self.height_m, self.edge)


def run_trial(
    cell: Cell,
    *,
    servo: str,
    rep: int,
    scenes: Path,
    official: Path,
    policies: str | None = None,
) -> dict[str, Any]:
    import mujoco
    import numpy as np

    from duckmind.skilltrain import dr as DR

    global _DUCK_CLASS
    if _DUCK_CLASS is None:
        _DUCK_CLASS = _make_duck_class()
    began = time.perf_counter()
    seed = seed_of(cell.key, rep)
    draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
    rng = np.random.default_rng(seed + 101)
    obstacle = cell.obstacle()
    folder = scene_dir(obstacle, scenes, official)
    aim = math.radians(float(rng.uniform(-AIM_DEG, AIM_DEG)))
    d0 = float(rng.uniform(*START_D))
    y0 = float(rng.uniform(-0.3, 0.3))
    duck = _DUCK_CLASS(
        [],
        servo=servo,
        draw=draw,
        head_alpha=1.0,
        keyframe="SIT",
        policies=policies,
        mjcf_dir=folder,
        x=FACE_X - 0.9,
        y=y0,
        yaw=float(rng.uniform(-math.pi, math.pi)),
    )
    m = duck.model
    obst_geoms = [
        g
        for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("hop_obst_")
    ]
    duck.setup_probe(obst_geoms)
    row: dict[str, Any] = {
        "key": f"{cell.key}/{servo}/{rep}",
        "cell": asdict(cell),
        "servo": servo,
        "rep": rep,
        "seed": seed,
        "dr": draw.to_json(),
        "aim_deg": round(math.degrees(aim), 3),
        "start_d_m": round(d0, 4),
        "excluded": None,
    }

    def excluded(why: str) -> dict[str, Any]:
        row["excluded"] = why
        row["wall_s"] = round(time.perf_counter() - began, 3)
        return row

    ok, _ = duck.stand(timeout=10.0, hold=1.0)
    if not ok:
        return excluded("did not stand up after the seated boot")
    duck.teleport(FACE_X - d0, y0, aim)
    for _ in range(int(float(rng.uniform(0.5, 1.0)) * CONTROL_HZ)):
        duck.move(0.0)
        duck.tick()
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return excluded("down before the walk")
    duck.touch_tick = False
    duck.robot_touch_any = False
    rec: dict[str, list[Any]] = {
        k: []
        for k in ("t", "tilt", "fallen", "trunk_x", "foot_x", "toe_hits", "housing", "tau_legs")
    }
    legs = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    t_walk = duck.now()
    t_touch: float | None = None
    t_cross: float | None = None
    t_walk_end: float | None = None
    t_stop_until: float | None = None
    touch_speed = None
    incidence = None
    yaw_hist: list[tuple[float, float]] = []
    far = obstacle.depth_m if obstacle is not None else 0.0
    feet = [int(g) for g in duck.foot_geoms]
    prev_tr_len = len(duck.trace)
    while True:
        now = duck.now()
        q = duck.data.qpos[duck.qadr + 3 : duck.qadr + 7]
        yaw = _yaw(q)
        yaw_hist.append((now, yaw))
        walking = t_walk_end is None
        if walking:
            vyaw = max(-HOLD_MAX, min(HOLD_MAX, -HOLD_K * _wrap(yaw - aim)))
            duck.move(cell.speed, 0.0, vyaw)
        else:
            duck.move(0.0)
        duck.toe_tick = 0
        duck.tick()
        now = duck.now()
        d = duck.data
        trunk_x = float(d.qpos[duck.qadr]) - FACE_X
        fx = [float(d.geom_xpos[g][0]) - FACE_X for g in feet]
        tr = duck.trace
        rec["t"].append(now)
        rec["tilt"].append(float(tr.data["tilt"][-1]))
        rec["fallen"].append(float(tr.data["fallen"][-1]))
        rec["trunk_x"].append(trunk_x)
        rec["foot_x"].append(fx)
        rec["toe_hits"].append(duck.toe_tick)
        rec["housing"].append(
            float(np.max(tr.data["f_housing"][-1])) if len(tr.data["f_housing"][-1]) else 0.0
        )
        rec["tau_legs"].append([float(tr.data["tau_pk"][-1][i]) for i in legs])
        prev_tr_len = len(tr)
        if obstacle is None:
            # flat control: a virtual face at x = 0 (the trunk centre crossing it)
            if t_touch is None and trunk_x >= -0.05:
                t_touch = now
        elif t_touch is None and duck.robot_touch_any:
            t_touch = now
        if t_touch is not None and touch_speed is None:
            v = d.qvel[duck.vadr : duck.vadr + 2]
            touch_speed = float(math.hypot(float(v[0]), float(v[1])))
            before = [y for (tt, y) in yaw_hist if tt <= now - 0.3]
            incidence = math.degrees(_wrap(before[-1] if before else yaw))
        if walking:
            if t_touch is None and now - t_walk > NO_TOUCH_S:
                return excluded("no touch within 10 s of walking")
            if t_cross is None and min(fx) > far + PAST_FOOT_M and trunk_x > far + PAST_TRUNK_M:
                t_cross = now
            if duck.safety.fallen:
                t_walk_end = now
                t_stop_until = now + WATCH_FALL_S
            elif t_cross is not None and now >= t_cross + AFTER_CROSS_S:
                t_walk_end = now
                t_stop_until = now + WATCH_S
            elif t_touch is not None and now >= t_touch + AFTER_TOUCH_S:
                t_walk_end = now
                t_stop_until = now + WATCH_S
        elif t_stop_until is not None and now >= t_stop_until:
            break
    del prev_tr_len
    rec_all = {**rec, "t_touch": t_touch, "t_walk": t_walk, "t_walk_end": t_walk_end}
    row.update(classify(rec_all, obstacle))
    row["touch_speed_mps"] = None if touch_speed is None else round(touch_speed, 4)
    row["incidence_deg"] = None if incidence is None else round(incidence, 2)
    row["walk_to_touch_s"] = None if t_touch is None else round(t_touch - t_walk, 3)
    row["sim_s"] = round(duck.now(), 3)
    row["wall_s"] = round(time.perf_counter() - began, 3)
    row["servo_info"] = duck.servo.info()
    return row


# ── the runner ──────────────────────────────────────────────────────────────────────────────


def cells(heights: Sequence[float] = HEIGHTS_M, flat: bool = True) -> list[Cell]:
    out = [Cell(h, e, v) for h in heights for e in EDGES for v in SPEEDS]
    if flat:
        out += [Cell(0.0, "sharp", v) for v in SPEEDS]
    return out


def _done(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    return rows


def run(args: argparse.Namespace) -> int:
    from duckmind.skilltrain.gate import Guard
    from duckmind.world.headless_duck import official_dir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    official = official_dir(None)
    scenes = out / "scenes"
    path = out / f"rows-{args.servo}.jsonl"
    done = _done(path)
    guard = Guard(not args.no_guard, log=lambda s: print(s, flush=True))
    heights = [float(h) for h in args.heights.split(",")] if args.heights else list(HEIGHTS_M)
    todo = cells(heights, flat=not args.no_flat)
    began = time.perf_counter()
    count = 0
    with path.open("a") as f:
        for cell in todo:
            valid = sum(
                1
                for r in done.values()
                if r["cell"] == asdict(cell) and r["servo"] == args.servo and not r["excluded"]
            )
            rep = 0
            while valid < args.n and rep < 3 * args.n:
                key = f"{cell.key}/{args.servo}/{rep}"
                if key in done:
                    rep += 1
                    continue
                guard.wait()
                row = run_trial(cell, servo=args.servo, rep=rep, scenes=scenes, official=official)
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[key] = row
                count += 1
                if not row["excluded"]:
                    valid += 1
                rep += 1
            print(
                f"[{args.servo}] {cell.key}: {valid} valid; {count} trials, "
                f"{time.perf_counter() - began:.0f} s",
                flush=True,
            )
    meta = {
        "servo": args.servo,
        "n": args.n,
        "trials_this_run": count,
        "wall_s": round(time.perf_counter() - began, 1),
        "guard_pauses": guard.pauses,
        "realtime_min": min(guard.readings, default=None),
        "official": str(official),
    }
    (out / f"meta-{args.servo}.json").write_text(json.dumps(meta, indent=1))
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": round(100 * k / n, 1) if n else None,
            "wilson95": [round(100 * lo, 1), round(100 * hi, 1)]}  # fmt: skip


def summarize(args: argparse.Namespace) -> int:
    import numpy as np

    out = Path(args.out)
    rows: list[dict[str, Any]] = []
    for servo in ("mjcf", "bam"):
        rows += list(_done(out / f"rows-{servo}.jsonl").values())
    groups: dict[tuple, list[dict[str, Any]]] = {}
    excluded: dict[str, int] = {}
    for r in rows:
        if r["excluded"]:
            excluded[r["excluded"]] = excluded.get(r["excluded"], 0) + 1
            continue
        c = r["cell"]
        groups.setdefault((r["servo"], c["edge"], c["height_m"], c["speed"]), []).append(r)
    summary: dict[str, Any] = {"cells": [], "excluded": excluded, "by_height": []}
    outcomes = ("crossed", "blocked", "stuck", "fall")
    for (servo, edge, h, v), rs in sorted(groups.items()):
        n = len(rs)
        cell = {"servo": servo, "edge": edge, "height_mm": round(h * 1000), "vx": v, "n": n}
        for o in outcomes:
            cell[o] = _rate(sum(1 for r in rs if r["outcome"] == o), n)
        cell["crossed_clean"] = _rate(
            sum(1 for r in rs if r["outcome"] == "crossed" and not r["stumble"]), n
        )
        cell["stumble"] = _rate(sum(1 for r in rs if r["stumble"]), n)
        cell["toe_hit_any"] = _rate(sum(1 for r in rs if r["toe_hits"] > 0), n)
        cell["housing_impact_any"] = _rate(sum(1 for r in rs if r["housing_impact_ticks"] > 0), n)
        tau = [r["leg_tau_p95"] for r in rs if r["leg_tau_p95"] is not None]
        cell["leg_tau_p95_median"] = round(float(np.median(tau)), 3) if tau else None
        cell["leg_tau_max_p90"] = round(float(np.percentile([r["leg_tau_max"] for r in rs], 90)), 3)
        tc = [r["t_cross_after_touch"] for r in rs if r["t_cross_after_touch"] is not None]
        cell["t_cross_median_s"] = round(float(np.median(tc)), 2) if tc else None
        sp = [r["touch_speed_mps"] for r in rs if r["touch_speed_mps"] is not None]
        cell["touch_speed_median"] = round(float(np.median(sp)), 3) if sp else None
        inc = [abs(r["incidence_deg"]) for r in rs if r["incidence_deg"] is not None]
        cell["incidence_abs_median_deg"] = round(float(np.median(inc)), 1) if inc else None
        summary["cells"].append(cell)
    # pooled over speeds
    pooled: dict[tuple, list[dict[str, Any]]] = {}
    for (servo, edge, h, _v), rs in groups.items():
        pooled.setdefault((servo, edge, h), []).extend(rs)
    for (servo, edge, h), rs in sorted(pooled.items()):
        n = len(rs)
        cell = {"servo": servo, "edge": edge, "height_mm": round(h * 1000), "n": n}
        for o in outcomes:
            cell[o] = _rate(sum(1 for r in rs if r["outcome"] == o), n)
        cell["crossed_clean"] = _rate(
            sum(1 for r in rs if r["outcome"] == "crossed" and not r["stumble"]), n
        )
        cell["stumble"] = _rate(sum(1 for r in rs if r["stumble"]), n)
        summary["by_height"].append(cell)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = ["| servo | edge | h mm | vx | n | crossed | clean | blocked | stuck | fall | stumble |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip

    def fmt(x: dict[str, Any]) -> str:
        return f"{x['k']}/{x['n']} ({x['wilson95'][0]:.0f}–{x['wilson95'][1]:.0f}%)"

    for c in summary["cells"]:
        lines.append(
            f"| {c['servo']} | {c['edge']} | {c['height_mm']} | {c['vx']:.2f} | {c['n']} | "
            f"{fmt(c['crossed'])} | {fmt(c['crossed_clean'])} | {fmt(c['blocked'])} | "
            f"{fmt(c['stuck'])} | {fmt(c['fall'])} | {fmt(c['stumble'])} |"
        )
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hop_probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--out", required=True)
    r.add_argument("--heights", default="")
    r.add_argument("--no-flat", action="store_true")
    r.add_argument("--no-guard", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
