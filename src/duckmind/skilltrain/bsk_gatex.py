"""b:gate: the full Mac gate of the B-list skills — every trained export and every "existing policy" choice.

The b:train gate (:mod:`bsk_gate` over :func:`bsk_probe.trained_cells`) asks one question per cell: did the
skill do its thing from its training entry. The owner's gate asks more (the workflow's b:gate list, 2026-09-28):

* **entries from walking and standing** for every skill (the mats and the bow were trained from standing
  only: their walking entries say what the brain's guard must do first — stop — rather than fail the skill);
* the **hand-back** as four outcomes: ``success`` / ``refused`` (robotd did not run it) / ``tripped`` (ran,
  did not finish, no fall: a stumble, a stop short, too slow, too wide) / ``fell`` (robotd's fall rule, from
  the request to the end of the trial);
* **righting after falls**: after any fall the trial runs on until 8 s past it (velstand owns righting; the
  brain's guard sends nothing to a fallen duck) — up again = tilt < 25°, no fall verdict, walk/stand, 1 s;
* **servo strain** (leg torque p95 per trial; BAM judged, the XML servos reported) and **time**;
* **the mat**: 3 / 4 / 5 cm, firm and soft tops, the training's approach (7–14 cm, ±8°) plus fixed angles
  0°, 15°, 25°, a walking entry (the request while still walking) and a walk-then-stop entry;
* **turning on the spot**: turn rate and the ground it needs (the largest distance from the start before
  90° of turn: "footprint") next to **the tightest v5 arc** (vx 0.35 + yaw 1, the smallest arc above the
  gait's dead zone) on the same ducks;
* **the bow**: the lowest front height (the beak tip) and return-to-stand.

Two sets, same ducks, same columns:

* ``trained`` — the B entries through robotd's ``[[policy.skill]]`` path (``--onnx-dir``: bskstep /
  bskbow / bskmat.onnx), released as the neural duck releases them (``skill_abort``, ``bow_release``);
* ``today`` — what the brain can use without training (b:feasibility's brain-only options): a 0.6 s nudge
  then a pure yaw (from walking: the pure yaw at once), the tightest arc, v5 at vx 0.20 (the march),
  robotd's ground pick (the deep bow), sit + head down (the low rest), walking onto / off the mat at 0.40.

Columns: the servo model (``mjcf`` = the owner's simulator's XML servos, ``bam`` = the real XL330's model with
the recipe's DR) × a fixed servo command delay (``--delay-steps`` 0 / 4 / 8 = 0 / 20 / 40 ms). Every trial is a
fresh DR duck (``bsk_probe``'s duck and control path; the same seed = the same physical duck on both models and
at every delay). ``verdict`` re-scores the rows (no physics) with Stage 1's levels (:func:`bsk_gate.level`).

    python -m duckmind.skilltrain.bsk_gatex run --set trained --onnx-dir DIR --servo mjcf --n 30 \\
        --delay-steps 4 --out GATE/d4
    python -m duckmind.skilltrain.bsk_gatex run --set today --servo bam --n 30 --delay-steps 0 --out TODAY/d0
    python -m duckmind.skilltrain.bsk_gatex verdict --gate GATE [--today TODAY]
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import bsk_gate as BG
from duckmind.skilltrain import bsk_probe as BP

CONTROL_HZ = BP.CONTROL_HZ
DT = BP.DT
RIGHTING_S, UPRIGHT_HOLD_S, UPRIGHT_DEG = 8.0, 1.0, 25.0
RETURN_DZ_MM, RETURN_HOLD_S = 10.0, 0.5
WALK_V = 0.40  # the entries' walking speed (Stage 1's walk0.40 posture)
MAT_WALK_V = 0.30  # walking up to a mat edge
MAT_WALK_FROM = (0.30, 0.40)  # trunk centre → the edge where that walk starts
STOP_COAST_M = 0.035  # a 0.30 walk told to stop still covers 3–4 cm (smoke, 2026-09-28)
STOP_SETTLE_S = (
    0.8  # walk-then-stop entries: standing this long before the request (the guard's settle)
)
REFUSE_S = 0.3  # a request whose skill label has not appeared by then was refused
AIMS_DEG = (
    0.0,
    15.0,
    25.0,
)  # fixed approach angles (the sign drawn per duck); None = the training's ±8°
HEIGHTS_MM = (30, 40, 50)
BEHAVIOUR = {  # cell name → the contract's gate family (bsk_contract.GATES)
    "turn_left": "turn", "turn_right": "turn", "march": "march", "step_up": "step_up",
    "step_down": "step_down", "bow": "bow", "lie_low": "lie",
    "pivot": "turn", "arc": "arc", "march_v5": "march", "ground_pick": "bow", "sit_head_down": "lie",
    "walk_on": "step_up", "walk_off": "step_down",
}  # fmt: skip
OUTCOMES = ("success", "refused", "tripped", "fell")


@dataclass(frozen=True)
class GCell:
    set: str  # trained | today
    name: str  # a B entry (trained) or a today option (BEHAVIOUR)
    entry: str  # stand | walk | walkstop
    height_mm: int = 0
    soft: bool = False
    aim_deg: float | None = None  # mats: None = the training's approach (±8°, 7–14 cm)

    @property
    def behaviour(self) -> str:
        return BEHAVIOUR[self.name]

    @property
    def is_mat(self) -> bool:
        return self.behaviour in ("step_up", "step_down")

    @property
    def key(self) -> str:
        k = f"{self.set}/{self.name}/{self.entry}"
        if self.is_mat:
            k += f"/{self.height_mm}mm{'-soft' if self.soft else ''}"
            if self.aim_deg is not None:
                k += f"/aim{self.aim_deg:g}"
        return k

    @property
    def core(self) -> bool:
        """A cell of the skill's training envelope (its verdict), else the guard's envelope (allowed or not)."""
        if self.behaviour in ("turn", "march", "arc"):
            return self.entry in ("stand", "walk")
        if self.is_mat:  # today's only mat entry is walking at it
            return self.set == "today" or (self.entry == "stand" and self.aim_deg in (None, 0.0))
        return (
            self.entry == "stand"
        )  # bow / lie: from standing (walk / walk-then-stop = the guard's envelope)


def trained_cells() -> list[GCell]:
    out = [
        GCell("trained", n, e)
        for n in ("turn_left", "turn_right", "march")
        for e in ("stand", "walk")
    ]
    out += [
        GCell("trained", n, e) for n in ("bow", "lie_low") for e in ("stand", "walk", "walkstop")
    ]
    for n in ("step_up", "step_down"):
        out += [GCell("trained", n, "stand", h, s) for h in HEIGHTS_MM for s in (False, True)]
        out += [GCell("trained", n, "stand", 40, False, a) for a in AIMS_DEG]
        out += [GCell("trained", n, e, h) for e in ("walk", "walkstop") for h in (30, 40)]
    return out


def today_cells() -> list[GCell]:
    out = [GCell("today", "pivot", e) for e in ("stand", "walk")]
    out += [GCell("today", "arc", "stand"), GCell("today", "arc", "walk")]
    out += [GCell("today", "march_v5", e) for e in ("stand", "walk")]
    out += [
        GCell("today", n, e)
        for n in ("ground_pick", "sit_head_down")
        for e in ("stand", "walk", "walkstop")
    ]
    out += [
        GCell("today", n, "walk", h, s)
        for n in ("walk_on", "walk_off")
        for h in HEIGHTS_MM
        for s in (False, True)
    ]
    return out


def seed_of(cell: GCell, rep: int) -> int:
    """Independent ducks per cell and repeat; the same duck on both servo models and at every delay."""
    body = cell.key.split("/", 1)[1]
    return zlib.crc32(f"bsk-g/{body}/{rep}".encode()) & 0x7FFFFFFF


# ── one trial ────────────────────────────────────────────────────────────────────────────────


def _labels_after(duck: Any, t0: float, t1: float) -> list[str]:
    import numpy as np

    tr = duck.trace.arrays()
    sel = (tr["t"] > t0) & (tr["t"] <= t1)
    return [str(x) for x in np.unique(tr["label"][sel])]


def _ran(duck: Any, names: Sequence[str], t0: float) -> bool:
    labs = _labels_after(duck, t0, t0 + REFUSE_S + 1e-9)
    return any(lab.split(":")[0] in names for lab in labs)


def _walk_until(duck: Any, seconds: float, v: float = WALK_V) -> None:
    BP._run(duck, seconds, lambda _t: duck.move(v))


def _held(duck: Any, name: str, seconds: float) -> dict[str, Any]:
    """Re-request a chaining entry every 0.1 s as the neural duck does, and stop asking once the duck is down
    (the guard sends nothing to a fallen duck)."""
    last, asked, accepted = -1e9, 0, 0
    for _ in range(int(round(seconds * CONTROL_HZ))):
        if duck.safety.fallen:
            break
        if duck.now() - last >= BP.RE_REQUEST_S - 1e-9:
            r = duck.do(name)
            asked += 1
            accepted += bool(r.get("accepted"))
            last = duck.now()
        duck.move(0.0)
        duck.tick()
    return {"asked": asked, "accepted": accepted}


def _entry(duck: Any, cell: GCell, rng: Any) -> str | None:
    """Bring the duck to the cell's entry posture on the flat floor. Returns a reason to exclude, or None."""
    if cell.entry == "walk":
        _walk_until(duck, float(rng.uniform(3.0, 3.5)))
    elif cell.entry == "walkstop":
        _walk_until(duck, float(rng.uniform(3.0, 3.5)))
        BP._run(duck, STOP_SETTLE_S, lambda _t: duck.move(0.0))
    else:
        BP._run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return "down before the request"
    return None


def _turn_extra(duck: Any, t0: float, sign: float) -> dict[str, Any]:
    """Time to 90° of turn and the ground used until then (the largest distance from the start)."""
    import numpy as np

    tr, ex = BP._arrays(duck)
    t = tr["t"]
    i0 = int(np.searchsorted(t, t0)) - 1
    yaw = np.unwrap(ex[:, 6])
    turned = sign * np.degrees(yaw - yaw[i0])
    idx = np.flatnonzero((t > t0) & (turned >= 90.0))
    xy = tr["pos"][:, :2]
    if not len(idx):
        return {"t_to_90_s": None, "footprint_90_m": None}
    i90 = int(idx[0])
    exc = float(np.linalg.norm(xy[i0 : i90 + 1] - xy[i0], axis=1).max())
    return {"t_to_90_s": round(float(t[i90] - t0), 3), "footprint_90_m": round(exc, 4)}


def _return_to_stand(duck: Any, t0: float, t_end: float | None) -> dict[str, Any]:
    """Back to standing after a bow / lie: walk/stand, tilt < 25°, the trunk within 10 mm of its height before
    the request, held 0.5 s — first time after the hand-back (or after t0 if there was none)."""
    import numpy as np

    tr, _ = BP._arrays(duck)
    t = tr["t"]
    pre = np.flatnonzero(t <= t0)[-5:]
    z0 = float(np.median(tr["pos"][pre, 2]))
    ok = (
        (tr["tilt"] < UPRIGHT_DEG)
        & (tr["fallen"] < 0.5)
        & np.array([lab in ("walk", "stand") for lab in tr["label"]])
        & (np.abs(tr["pos"][:, 2] - z0) * 1000 <= RETURN_DZ_MM)
    )
    ok &= t > (t_end if t_end is not None else t0)
    need = int(round(RETURN_HOLD_S / DT))
    run = 0
    for i in range(len(t)):
        run = run + 1 if ok[i] else 0
        if run >= need:
            first = i - need + 1
            return {"returned": True, "returned_s": round(float(t[first] - t0), 3)}
    return {"returned": False, "returned_s": None}


def _righting(duck: Any, t0: float, stop_skill: str | None) -> dict[str, Any]:
    """robotd's fall rule from the request on; after a fall, run on (zero twist) to 8 s + 1 s past it and
    say when the duck was up again."""
    import numpy as np

    tr = duck.trace.arrays()
    t = tr["t"]
    fallen = (tr["fallen"] > 0.5) & (t >= t0)
    if not fallen.any():
        return {"fell_any": False, "righted_8s": None, "righted_s": None}
    tf = float(t[np.flatnonzero(fallen)[0]])
    need = tf + RIGHTING_S + UPRIGHT_HOLD_S + 0.1 - duck.now()
    if stop_skill:
        duck.do(stop_skill)
    if need > 0:
        BP._run(duck, need, lambda _t: duck.move(0.0))
    tr = duck.trace.arrays()
    t = tr["t"]
    i_f = int(np.searchsorted(t, tf))
    ok = (
        (tr["tilt"] < UPRIGHT_DEG)
        & (tr["fallen"] < 0.5)
        & np.array([lab in ("walk", "stand") for lab in tr["label"]])
    )
    hold = int(round(UPRIGHT_HOLD_S / DT))
    run = 0
    for i in range(i_f, len(t)):
        run = run + 1 if ok[i] else 0
        if run >= hold:
            s = float(t[i - hold + 1] - tf)
            return {"fell_any": True, "fall_t_s": round(tf - t0, 3), "righted_s": round(s, 3),
                    "righted_8s": bool(s <= RIGHTING_S)}  # fmt: skip
    return {"fell_any": True, "fall_t_s": round(tf - t0, 3), "righted_s": None, "righted_8s": False}


def _stepping_trial(duck: Any, cell: GCell, rng: Any) -> dict[str, Any]:
    """turn_left / turn_right / march (trained, held 4 s, released by skill_abort) and the today options
    pivot / arc / march_v5 (a twist for 4 s, then zero)."""
    why = None
    sign = 1.0
    if cell.set == "today" and cell.name == "pivot" and cell.entry == "stand":
        BP._run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        _walk_until(
            duck, 0.6
        )  # the nudge: velstand needs to be stepping before a pure yaw turns it
    else:
        why = _entry(duck, cell, rng)
    if why or duck.safety.fallen or duck.tilt_deg() > 30.0:
        return {"excluded": why or "down before the request"}
    t0 = duck.now()
    req: dict[str, Any] = {}
    if cell.set == "trained":
        sign = -1.0 if cell.name == "turn_right" else 1.0
        req = _held(duck, cell.name, BP.SKILL_HOLD_S[cell.name])
        t1 = duck.now()
        if not duck.safety.fallen:
            duck.do("skill_abort")
        refused = not _ran(duck, (cell.name,), t0)
    else:
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        vx, vyaw = {"pivot": (0.0, 1.0), "arc": (0.35, 1.0), "march_v5": (0.20, 0.0)}[cell.name]
        BP._run(duck, BP.TURN_S, lambda _t: duck.move(vx, 0.0, sign * vyaw))
        t1 = duck.now()
        refused = False
    BP._run(duck, BP.AFTER_S, lambda _t: duck.move(0.0))
    m = BP.gait_metrics(duck, t0, t1, sign)
    return {
        "t0": t0,
        "t1": t1,
        "sign": sign,
        "refused": refused,
        **req,
        **m,
        **_turn_extra(duck, t0, sign),
    }


def _bow_trial(duck: Any, cell: GCell, rng: Any) -> dict[str, Any]:
    """bow (one-shot) / lie_low (held 3 s, ended by bow_release); today: robotd's ground pick, or sit + head
    down 2 s and stand up."""
    why = _entry(duck, cell, rng)
    if why:
        return {"excluded": why}
    t0 = duck.now()
    t_end: float | None = None
    refused = False
    extra: dict[str, Any] = {}
    if cell.set == "trained":
        if cell.name == "lie_low":
            extra = _held(duck, "lie_low", BP.SKILL_HOLD_S["lie_low"])
            refused = not _ran(duck, ("lie_low",), t0)
            if not duck.safety.fallen:
                duck.do("bow_release")
            t_end = BP._until_label_leaves(duck, ("bow_release", "lie_low"), 3.0)
        else:
            r = duck.do("bow")
            t_end = BP._until_label_leaves(duck, ("bow",), 4.0)
            refused = not r.get("accepted") or not _ran(duck, ("bow",), t0)
        if t_end is None:
            t_end = duck.now()
            extra["no_handback"] = True
        BP._run(duck, BP.HANDBACK_S, lambda _t: duck.move(0.0))
        return {"t0": t0, "t_end": t_end, "refused": refused, **extra, **BP.bow_metrics(duck, t0, t_end),
                **_return_to_stand(duck, t0, t_end)}  # fmt: skip
    if cell.name == "ground_pick":
        duck.do("ground_pick")
        for _ in range(int(4.0 * CONTROL_HZ)):
            duck.move(0.0)
            duck.tick()
            if duck.policy != "ground_pick" and duck.now() - t0 > 0.1:
                t_end = duck.now()
                break
        refused = not _ran(duck, ("ground_pick",), t0)
        if t_end is None:
            t_end = duck.now()
            extra["no_handback"] = True
        BP._run(duck, BP.HANDBACK_S, lambda _t: duck.move(0.0))
        return {"t0": t0, "t_end": t_end, "refused": refused, **extra, **BP.bow_metrics(duck, t0, t_end),
                **_return_to_stand(duck, t0, t_end)}  # fmt: skip
    # sit + head down
    duck.do("sit_toggle")
    waited = 0.0
    while duck.policy != "sit" and waited < 6.0:
        duck.move(0.0)
        duck.tick()
        waited += DT
    if duck.policy != "sit":
        BP._run(duck, 2.0, lambda _t: duck.move(0.0))
        return {"t0": t0, "refused": True, "t_end": None, **BP.bow_metrics(duck, t0, duck.now()),
                **_return_to_stand(duck, t0, None)}  # fmt: skip
    BP._run(duck, 2.5)
    t_head = duck.now()
    BP._run(duck, 2.0, lambda _t: duck.head(BP.LIE_HEAD))
    t_head_end = duck.now()
    duck.head((0.0, 0.0, 0.0, 0.0))
    BP._run(duck, 0.8)
    t_rise = duck.now()
    duck.do("sit_toggle")
    BP._run(duck, 4.0, lambda _t: duck.move(0.0))
    lie = BP.lie_metrics(duck, t0, t_head, t_head_end, t_rise)
    bowm = BP.bow_metrics(duck, t0, t_rise)
    return {"t0": t0, "t_end": t_rise, "refused": False, **bowm, **{f"sit_{k}": v for k, v in lie.items()},
            **_return_to_stand(duck, t0, t_rise)}  # fmt: skip


def _mat_trial(duck: Any, cell: GCell, rng: Any) -> dict[str, Any]:
    """step_up / step_down (trained) from the cell's entry and approach angle; today: walk on / off at 0.40."""
    import numpy as np

    h = cell.height_mm / 1000.0
    up = cell.behaviour == "step_up"
    if cell.set == "today":
        bp = BP.Cell("mat_up" if up else "mat_dn", "walk", 0.40, 0.0, h, cell.soft)
        t0 = duck.now()
        res = BP._mat_trial(duck, bp, rng)
        if res.get("excluded"):
            return res
        return {"t0": t0, "refused": False, **res}
    aim_deg = float(rng.uniform(-B.MAT_AIM_DEG, B.MAT_AIM_DEG)) if cell.aim_deg is None else (
        cell.aim_deg * (1.0 if rng.uniform() < 0.5 else -1.0))  # fmt: skip
    aim = math.radians(aim_deg)
    y0 = float(rng.uniform(-0.3, 0.3))
    d = float(rng.uniform(*B.MAT_START_D))
    edge_x = BP.MAT_FACE_X + (BP.MAT_DEPTH_M if not up else 0.0)
    if cell.entry == "stand":
        duck.teleport(edge_x - d, y0, aim)
        BP._run(duck, float(rng.uniform(0.5, 1.0)), lambda _t: duck.move(0.0))
    else:
        start = float(rng.uniform(*MAT_WALK_FROM))
        if not up:
            start = min(start, BP.MAT_DEPTH_M - 0.06)  # on the mat, clear of its far edge
        duck.teleport(edge_x - start, y0, aim)
        BP._run(duck, float(rng.uniform(0.5, 1.0)), lambda _t: duck.move(0.0))
        goal = d + (STOP_COAST_M if cell.entry == "walkstop" else 0.0)
        for _ in range(int(6.0 * CONTROL_HZ)):
            err = BP._wrap(BP._yaw_of(duck) - aim)
            duck.move(MAT_WALK_V, 0.0, max(-BP.HOLD_MAX, min(BP.HOLD_MAX, -BP.HOLD_K * err)))
            duck.tick()
            if edge_x - float(duck.data.qpos[duck.qadr]) <= goal or duck.safety.fallen:
                break
        if cell.entry == "walkstop":
            BP._run(duck, STOP_SETTLE_S, lambda _t: duck.move(0.0))
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return {"excluded": "down before the request"}
    dist = round(edge_x - float(duck.data.qpos[duck.qadr]), 4)
    t0 = duck.now()
    r = duck.do(cell.name)
    t_end = BP._until_label_leaves(duck, (cell.name,), 5.0)
    refused = not r.get("accepted") or not _ran(duck, (cell.name,), t0)
    BP._run(duck, 3.0, lambda _t: duck.move(0.0))
    ex = np.asarray(duck.extra[-5:], float)
    xl, xr, zl, zr = ex[-1, 2], ex[-1, 3], ex[-1, 4], ex[-1, 5]
    rest = BP._foot_rest_z(duck)
    past = min(xl, xr) > edge_x + BP.PAST_FOOT_M * 0.5
    over = past and (min(zl, zr) > rest + 0.5 * h if up else max(zl, zr) < rest + 0.5 * h)
    over = bool(over and duck.tilt_deg() < 45.0 and not duck.safety.fallen)
    bp = BP.Cell("mat_up" if up else "mat_dn", "stand", 0.0, 0.0, h, cell.soft, skill=cell.name)
    res = BP.mat_metrics(duck, bp, t0, t0, t_end if over else None, over, None)
    return {"t0": t0, "t_end": t_end, "refused": refused, "handed_back": t_end is not None, "aim_deg": round(aim_deg, 2),
            "dist_at_request_m": dist, **res}  # fmt: skip


def run_trial(cell: GCell, *, servo: str, rep: int, scenes: Path, delay_steps: int | None,
              onnx_dir: str | None = None, policies: str | None = None) -> dict[str, Any]:  # fmt: skip
    import numpy as np

    from duckmind.skilltrain import dr as DR
    from duckmind.world.headless_duck import official_dir, policy_dir

    began = time.perf_counter()
    seed = seed_of(cell, rep)
    draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
    rng = np.random.default_rng(seed + 11)
    folder = None
    if cell.is_mat:
        from duckmind.skilltrain.hop_probe import scene_dir

        bp = BP.Cell("mat_up", "stand", 0.0, 0.0, cell.height_mm / 1000.0, cell.soft)
        folder = scene_dir(BP._mat(bp), scenes, official_dir(None))
    Duck = BP._duck_class()
    pol = Path(policies) if policies else policy_dir()
    skills = BP.b_skilldefs(onnx_dir, pol) if cell.set == "trained" else []
    if cell.set == "trained" and cell.name not in {d.name for d in skills}:
        raise ValueError(f"{cell.name}: no ONNX for it in {onnx_dir}")
    duck = Duck(skills, servo=servo, draw=draw, head_alpha=1.0, keyframe="SIT", policies=policies,
                mjcf_dir=folder, x=0.0 if folder is None else BP.MAT_FACE_X - 0.9,
                y=0.0, yaw=float(rng.uniform(-math.pi, math.pi)))  # fmt: skip
    if delay_steps is not None:
        from collections import deque

        duck.lag_steps = int(delay_steps)
        duck._lag = deque(maxlen=duck.lag_steps + 1)
    if cell.set == "today" and cell.name == "ground_pick":
        duck.port = BP._ground_pick_port(pol / "velstand.onnx", pol / "alpha_sitstand.onnx",
                                         pol / "alpha_ground_pick.onnx", 1.0)  # fmt: skip
        duck.skill_log = duck.port.controller.events
    row: dict[str, Any] = {"key": f"{cell.key}/{servo}/{rep}", "cell": asdict(cell), "servo": servo, "rep": rep,
                           "seed": seed, "dr": draw.to_json(), "excluded": None}  # fmt: skip
    ok, _ = duck.stand(timeout=10.0, hold=1.0)
    if not ok:
        row["excluded"] = "did not stand up after the seated boot"
        return row
    ex0 = np.asarray(duck.extra[-10:], float)
    duck._rest_z = float(np.min(np.minimum(ex0[:, 4], ex0[:, 5])))
    row["delay_steps"] = int(duck.lag_steps)
    beh = cell.behaviour
    if beh in ("turn", "march", "arc"):
        res = _stepping_trial(duck, cell, rng)
    elif beh in ("bow", "lie"):
        res = _bow_trial(duck, cell, rng)
    else:
        res = _mat_trial(duck, cell, rng)
    if res.get("excluded"):
        row["excluded"] = res["excluded"]
        return row
    stop = "skill_abort" if cell.set == "trained" and beh in ("turn", "march") else None
    res.update(_righting(duck, float(res["t0"]), stop))
    row.update(res)
    row.update(judge(row))
    row["sim_s"] = round(duck.now(), 3)
    row["wall_s"] = round(time.perf_counter() - began, 3)
    return row


# ── scoring (pure) ───────────────────────────────────────────────────────────────────────────


def _num(r: Mapping[str, Any], k: str) -> float | None:
    v = r.get(k)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def checks(r: Mapping[str, Any]) -> dict[str, bool]:
    """The contract's checks (bsk_gate's, for the family), without the fall check (an outcome of its own)."""
    beh = BEHAVIOUR[r["cell"]["name"]]
    if beh == "arc":
        return {"turned_90_in_4s": (_num(r, "dyaw_deg") or 0.0) >= 90.0}
    shim_skill = {"turn": "turn_left", "march": "march", "step_up": "step_up", "step_down": "step_down",
                  "bow": "bow", "lie": "lie_low"}[beh]  # fmt: skip
    fam = {
        "turn": "turn",
        "march": "march",
        "step_up": "mat_up",
        "step_down": "mat_dn",
        "bow": "bow",
        "lie": "lie",
    }[beh]
    shim = {**r, "cell": {"family": fam, "skill": shim_skill}}
    if beh in ("step_up", "step_down") and r.get("outcome") == "fall":
        shim["outcome"] = "fall"
    c = BG.checks(shim)
    c.pop("no_fall", None)
    if beh in ("bow", "lie"):
        c["back_to_standing"] = bool(r.get("returned"))
    return c


def outcome(r: Mapping[str, Any]) -> str:
    if r.get("refused"):
        return "refused"
    if r.get("fell_any") or r.get("outcome") == "fall":
        return "fell"
    c = checks(r)
    if c and all(c.values()):
        return "success"
    return "tripped"


def judge(r: Mapping[str, Any]) -> dict[str, Any]:
    c = checks(r)
    return {"checks": c, "outcome_g": outcome(r), "missed": [k for k, v in c.items() if not v]}


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = B.wilson(k, n)
    return {
        "k": k,
        "n": n,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "text": f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f} %]",
    }


def _q(vals: Iterable[Any], nd: int = 3) -> dict[str, Any] | None:
    import numpy as np

    v = [float(x) for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None
    a = np.asarray(v)
    return {"p50": round(float(np.median(a)), nd), "p10": round(float(np.percentile(a, 10)), nd),
            "p90": round(float(np.percentile(a, 90)), nd), "n": len(v)}  # fmt: skip


def qtext(q: Mapping[str, Any] | None, unit: str = "", scale: float = 1.0, nd: int = 1) -> str:
    if not q:
        return "—"
    f = f"{{:.{nd}f}}"
    return f"{f.format(q['p50'] * scale)}{unit} ({f.format(q['p10'] * scale)}–{f.format(q['p90'] * scale)})"


MEASURES = {
    "turn": ("dyaw_deg", "rate_last2_dps", "t_to_90_s", "footprint_90_m", "excursion_m", "footfalls", "foot_lift_mm"),
    "arc": ("dyaw_deg", "rate_last2_dps", "t_to_90_s", "footprint_90_m", "excursion_m", "radius_m"),
    "march": ("footfalls", "foot_lift_mm", "net_m", "dyaw_deg"),
    "bow": ("pitch_down_max_deg", "beak_min_mm", "beak_drop_mm", "trunk_dip_mm", "t_bow_s", "hold_bow_s",
            "t_lowest_s", "duration_s", "returned_s", "footfalls"),
    "lie": ("pitch_down_max_deg", "beak_min_mm", "trunk_dip_mm", "hold_lie_s", "duration_s", "returned_s",
            "sit_seated_drop_mm", "sit_rise_s"),
    "step_up": ("t_done_after_touch", "max_tilt_deg", "dist_at_request_m", "aim_deg"),
    "step_down": ("t_done_after_touch", "max_tilt_deg", "dist_at_request_m", "aim_deg"),
}  # fmt: skip


def load(gate_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """{column: valid rows} from DIR/d0, d4, d8 (rows-mjcf / rows-bam); a key seen twice counts once (its last)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for d in BG.DELAYS:
        for servo in ("mjcf", "bam"):
            p = Path(gate_dir) / d / f"rows-{servo}.jsonl"
            if not p.is_file():
                continue
            seen: dict[str, dict[str, Any]] = {}
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("excluded") or not (r.get("cell") or {}).get("name"):
                    continue
                seen[r["key"]] = {**r, **judge(r)}  # re-scored with the current rules
            out[f"{servo}_{d}"] = list(seen.values())
    return out


def cell_block(rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rs)
    beh = BEHAVIOUR[rs[0]["cell"]["name"]]
    oc = {o: _rate(sum(r["outcome_g"] == o for r in rs), n) for o in OUTCOMES}
    falls = sum(r["outcome_g"] == "fell" for r in rs)
    perf = sum(r["outcome_g"] == "success" for r in rs)
    fallen = [r for r in rs if r.get("fell_any")]
    up = [r for r in fallen if r.get("righted_8s")]
    missed: dict[str, int] = {}
    for r in rs:
        if r["outcome_g"] == "tripped":
            for m in r.get("missed") or ["stumble"]:
                missed[m] = missed.get(m, 0) + 1
    return {
        "n": n,
        "level": BG.level(perf, falls, n) if beh != "arc" else "reference",
        "outcomes": oc,
        "tripped_why": missed,
        "righting": {
            "falls": len(fallen),
            "righted_8s": _rate(len(up), len(fallen)) if fallen else None,
            "righted_s": _q(r.get("righted_s") for r in up),
        },  # fmt: skip
        "leg_tau_p95": _q((r.get("leg_tau_p95") for r in rs), 4),
        "head_tau_p95": _q((r.get("head_tau_p95") for r in rs), 4),
        "measures": {k: _q(r.get(k) for r in rs) for k in MEASURES.get(beh, ())},
    }


def score(columns: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for col, rows in columns.items():
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            groups.setdefault(GCell(**r["cell"]).key, []).append(r)
        for key, rs in groups.items():
            c = cells.setdefault(key, {"cell": rs[0]["cell"], "columns": {}})
            c["columns"][col] = cell_block(rs)
    skills: dict[str, dict[str, Any]] = {}
    for key, c in sorted(cells.items()):
        g = GCell(**c["cell"])
        cols = c["columns"]
        nominal = cols.get(BG.NOMINAL, {}).get("level", "none")
        dr = {k: v["level"] for k, v in cols.items() if k != BG.NOMINAL}
        bam = [
            v["leg_tau_p95"]["p50"]
            for k, v in cols.items()
            if k.startswith("bam") and v["leg_tau_p95"]
        ]
        mj = [
            v["leg_tau_p95"]["p50"]
            for k, v in cols.items()
            if k.startswith("mjcf") and v["leg_tau_p95"]
        ]
        worst = max(bam) if bam else None
        c["strain_bam_nm"], c["strain_mjcf_nm"] = worst, (max(mj) if mj else None)
        c["strain_bam"] = (None if worst is None else "ok" if worst <= BG.STRAIN_OK_NM else
                           "marginal" if worst <= BG.STRAIN_MAX_NM else "over")  # fmt: skip
        c["nominal"], c["dr_levels"] = nominal, dr
        c["pass"] = (
            nominal == "ok" and len(dr) >= 5 and all(v in ("ok", "weak") for v in dr.values())
        )
        c["pass_limits"] = nominal in ("ok", "weak")
        c["core"] = g.core
        if g.behaviour == "arc":
            continue
        s = skills.setdefault(f"{g.set}/{g.name}", {"core": [], "envelope": [], "limits": []})
        (s["core"] if g.core else s["envelope"]).append(key)
        if g.core and not c["pass"]:
            bad = [f"{k} {v}" for k, v in {BG.NOMINAL: nominal, **dr}.items() if v != "ok"]
            s["limits"].append(f"{key}: " + ", ".join(bad))
    for name, s in skills.items():
        cs = [cells[k] for k in s["core"]]
        s["verdict"] = ("pass" if cs and all(c["pass"] for c in cs) else "pass with limits"
                        if cs and all(c["pass_limits"] for c in cs) else "fail")  # fmt: skip
        s["allowed"] = [k for k in s["core"] + s["envelope"] if cells[k]["pass"]]
        s["not_allowed"] = [k for k in s["core"] + s["envelope"] if not cells[k]["pass"]]
        s["strain_bam"] = max((cells[k]["strain_bam"] or "ok" for k in s["core"] + s["envelope"]),
                              key=("ok", "marginal", "over").index)  # fmt: skip
    return {
        "format": "duckmind-bsk-gatex-1",
        "nominal": BG.NOMINAL,
        "cells": cells,
        "skills": skills,
    }


COLS = ("mjcf_d0", "mjcf_d4", "mjcf_d8", "bam_d0", "bam_d4", "bam_d8")


def report(res: Mapping[str, Any], title: str) -> str:
    lines = [f"# {title}", "",
             "Columns: servo model (mjcf = the owner's simulator's XML servos, bam = the XL330 model with DR) × servo "
             "command delay (d0/d4/d8 = 0/20/40 ms). Nominal = mjcf_d4. Each cell: level · success · fell "
             "(k/n [Wilson 95 %], independent DR ducks). Outcomes: success / refused / tripped / fell.", ""]  # fmt: skip
    lines.append("| cell | " + " | ".join(COLS) + " | strain p95 bam / mjcf (N·m) |")
    lines.append("|---|" + "---|" * len(COLS) + "---|")
    for key, c in sorted(res["cells"].items()):
        row = []
        for col in COLS:
            v = c["columns"].get(col)
            if v is None:
                row.append("—")
                continue
            o = v["outcomes"]
            row.append(f"{v['level']} · ok {o['success']['text']} · fell {o['fell']['text']}"
                       + (f" · refused {o['refused']['k']}" if o["refused"]["k"] else ""))  # fmt: skip
        lines.append(
            f"| {key} | " + " | ".join(row) + f" | {c['strain_bam_nm']} / {c['strain_mjcf_nm']} |"
        )
    lines.append("")
    for name, s in sorted(res["skills"].items()):
        lines.append(f"- **{name}**: {s['verdict']} (strain on BAM {s['strain_bam']}); allowed cells "
                     f"{len(s['allowed'])}/{len(s['allowed']) + len(s['not_allowed'])}")  # fmt: skip
        for lim in s["limits"]:
            lines.append(f"  - {lim}")
    lines.append("")
    lines.append("## Details per cell (mjcf_d4 and bam_d4; p50 (p10–p90))")
    for key, c in sorted(res["cells"].items()):
        for col in ("mjcf_d4", "bam_d4"):
            v = c["columns"].get(col)
            if not v:
                continue
            ms = "; ".join(f"{k} {qtext(q, nd=3)}" for k, q in v["measures"].items() if q)
            rt = v["righting"]
            rtt = f"righted ≤ 8 s {rt['righted_8s']['text']}" if rt["righted_8s"] else "no falls"
            lines.append(f"- {key} · {col}: tripped {v['outcomes']['tripped']['text']} {v['tripped_why'] or ''}; "
                         f"{rtt}; leg τ p95 {qtext(v['leg_tau_p95'], nd=3)}; {ms}")  # fmt: skip
    return "\n".join(lines) + "\n"


# ── runner ───────────────────────────────────────────────────────────────────────────────────


def _done(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                rows[r["key"]] = r
            except (ValueError, KeyError):
                continue
    return rows


def run(args: argparse.Namespace) -> int:
    from duckmind.skilltrain.gate import Guard

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = out.parent / "scenes"
    path = out / f"rows-{args.servo}.jsonl"
    done = _done(path)
    guard = Guard(not args.no_guard, log=lambda s: print(s, flush=True))
    pool = trained_cells() if args.set == "trained" else today_cells()
    if args.set == "trained":
        have = {p.stem for p in Path(args.onnx_dir).glob("*.onnx")}
        nets = {e.name: e.net for e in B.ENTRIES}
        pool = [c for c in pool if nets[c.name] in have]
    todo = [
        c for c in pool if not args.only or any(c.key.startswith(p) for p in args.only.split(","))
    ]
    began = time.perf_counter()
    count = 0
    with path.open("a") as f:
        for cell in todo:
            valid = sum(1 for r in done.values() if r["cell"] == asdict(cell) and not r["excluded"])
            rep = 0
            while valid < args.n and rep < 3 * args.n:
                key = f"{cell.key}/{args.servo}/{rep}"
                if key in done:
                    rep += 1
                    continue
                guard.wait()
                try:
                    row = run_trial(cell, servo=args.servo, rep=rep, scenes=scenes, delay_steps=args.delay_steps,
                                    onnx_dir=args.onnx_dir)  # fmt: skip
                except Exception as exc:  # noqa: BLE001 - record and go on
                    row = {"key": key, "cell": asdict(cell), "servo": args.servo, "rep": rep,
                           "excluded": f"error: {type(exc).__name__}: {exc}"}  # fmt: skip
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[key] = row
                count += 1
                valid += not row["excluded"]
                rep += 1
            print(f"[{args.servo} d{args.delay_steps}] {cell.key}: {valid} valid; {count} trials, "
                  f"{time.perf_counter() - began:.0f} s", flush=True)  # fmt: skip
    meta = {"servo": args.servo, "set": args.set, "n": args.n, "delay_steps": args.delay_steps,
            "onnx_dir": args.onnx_dir, "trials_this_run": count, "wall_s": round(time.perf_counter() - began, 1),
            "guard_pauses": guard.pauses, "realtime_min": min(guard.readings, default=None)}  # fmt: skip
    (out / f"meta-{args.servo}.json").write_text(json.dumps(meta, indent=1))
    return 0


def verdict(args: argparse.Namespace) -> int:
    res = score(load(args.gate))
    Path(args.gate, "verdict.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    text = report(res, args.title or f"B-skills gate: {Path(args.gate).name}")
    Path(args.gate, "verdict.md").write_text(text)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_gatex")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", choices=("trained", "today"), required=True)
    r.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--out", required=True)
    r.add_argument("--only", default="")
    r.add_argument("--no-guard", action="store_true")
    r.add_argument("--onnx-dir", default=None)
    r.add_argument("--delay-steps", type=int, required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--gate", required=True)
    v.add_argument("--title", default="")
    a = ap.parse_args(argv)
    if a.cmd == "run" and a.set == "trained" and not a.onnx_dir:
        ap.error("--set trained needs --onnx-dir")
    return run(a) if a.cmd == "run" else verdict(a)


if __name__ == "__main__":
    raise SystemExit(main())
