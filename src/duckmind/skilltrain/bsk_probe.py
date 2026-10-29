"""B-list skills, feasibility part (a): what the shipped v5 set and robotd already do (b:feasibility).

Owner 2026-09-28 「A B 全數」: four trained movement skills — (1) 原地轉身 turning on the spot, used only
where needed (normal turns stay arcs), (2) 原地踏步／跺腳 marching / stomping in place, (3) stepping up
onto a 3–5 cm mat and back down, (4) 鞠躬／趴低 bowing / lying low with the front down and back up.
Before any unit is spent: what the official policies do today for each, on both servo models of the
Stage-1 gates (``mjcf``: the MJCF's XML servos, the owner's simulator, 0 ms command delay; ``bam``:
training's BAM XL330 m6 with the recipe's DR incl. its 15–30 ms command delay), independent DR ducks
(:mod:`duckmind.skilltrain.dr`, the same seed = the same physical duck on both models), n ≥ 30 per cell.

The duck is the Stage-1 gate duck (:class:`~duckmind.skilltrain.skillduck.SkillDuck`: the official
``scene_allcollisions.xml``, robotd's control path ported line by line) with robotd's **ground pick**
added (``control.rs``: the phase encoding ``[cos 2πφ, sin 2πφ, 0]``, φ += dt / 4 s, hand-back at
φ = 0.7, action scale 1.0 × voltage adaptation, gain ratio 1.0, priority skill > ground pick > sit),
because the v5 set's ``alpha_ground_pick.onnx`` is the one upstream front-down motion. The mats reuse
the hop workflow's terrain (:class:`duckmind.skilltrain.hop_probe.Obstacle` / ``scene_dir``: a strip
across the official floor; a mat is the same strip 0.40 m deep), with an optional soft top (a
compliant ``solref``) for 「軟墊」.

Cells (:func:`cells`), each trial = a fresh DR duck booting seated, rising, then the entry:

======  =====  ======================================  ==============================================
family  entry  command                                 question
======  =====  ======================================  ==============================================
turn    stand  (0, 0, ±1) 4 s                          does v5 turn on the spot from standing?
turn    walk   (0, 0, ±1) 4 s after a 0.40 walk        …once it is already stepping?
turn    walk   (0.20, 0, ±1)                           a sub-threshold arc after a walk
turn    walk   (0.30, 0, ±1)                           Duck Core's arc today
turn    stand  (0.35, 0, ±1)                           the tightest arc above the dead zone
march   stand  (0.20, 0, 0) 4 s                        does v5 step in place in the dead zone?
march   walk   (0.20, 0, 0) 4 s after a 0.40 walk      …or keep stepping once walking?
bow     stand  robot.do ground_pick (2.8 s), 5 s       the upstream front-down reach, hand-back to v5
lie     stand  sit_toggle, head down 2 s, stand up     the upstream "low rest" (sitstand) + head down
mat_up  walk   0.40 at a 3 / 5 cm mat edge             (the hop probe's 4 cm strips: same first contact)
mat_dn  stand  on a 3 / 4 / 5 cm mat, walk off 0.40    stepping down (not in the hop probe)
soft    …      the 4 cm mat with a compliant top       up and down
======  =====  ======================================  ==============================================

Truth only (evaluators): trunk pose, yaw, footfalls (a re-contact after ≥ 40 ms of air, phase 1's),
foot lift (foot geom height above its start), the beak tip's height, robotd's fall rule, peak actuator
force per tick, XL330-housing floor forces.

    python -m duckmind.skilltrain.bsk_probe run --servo mjcf --n 30 --out DIR
    python -m duckmind.skilltrain.bsk_probe summarize --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
LEGS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
HEAD = (5, 6, 7, 8)
TORQUE_NM = 0.4
STALL_VEL = 0.5
HOUSING_N = 2.0
GP_PERIOD_S, GP_END_PHASE = 4.0, 0.7  # robotd control.rs defaults (manifest: duration 2.8 s)
TURN_S = 4.0
AFTER_S = 2.0
HANDBACK_S = 5.0
MAT_DEPTH_M = 0.40
MAT_FACE_X = 1.0
SOFT_SOLREF = (0.06, 1.0)  # a compliant top: 7.3 mm of sink under the standing duck (mjcf, nominal)
SOFT_SOLIMP = (0.4, 0.95, 0.015, 0.5, 2.0)
SOFT_FRICTION = (0.9, 0.005, 0.0001)  # fabric; priority 2 so the mat's own contact rules the feet's
LIE_HEAD = (-0.3, 0.6, 0.0, 0.0)  # robot.head offsets: neck forward/down, beak down
PAST_FOOT_M, PAST_TRUNK_M = 0.028, 0.03  # hop_probe's "past the face" margins (a sole is 5.4 cm)
STUMBLE_DEG, STUCK_DEG, STUCK_S = 20.0, 40.0, 1.0
HOLD_K, HOLD_MAX = 3.0, 1.0  # heading hold while walking at a mat (hop_probe's)


@dataclass(frozen=True)
class Cell:
    family: str  # turn | march | bow | lie | mat_up | mat_dn
    entry: str  # stand | walk
    vx: float = 0.0
    vyaw: float = 0.0
    height_m: float = 0.0
    soft: bool = False
    pre_s: float = (
        0.0  # entry "nudge": a 0.40 walk this long before the command (a cornered duck's room)
    )
    skill: str | None = None  # a trained B entry (bsk_contract) instead of a v5 twist / ground pick

    @property
    def key(self) -> str:
        if self.skill:
            where = (
                f"{round(self.height_m * 1000):d}mm{'-soft' if self.soft else ''}"
                if self.family.startswith("mat")
                else self.entry
            )
            return f"skill/{self.skill}/{where}"
        if self.family in ("turn", "march"):
            pre = f"{self.pre_s:.1f}s" if self.entry == "nudge" else ""
            return f"{self.family}/{self.entry}{pre}/vx{self.vx:.2f}/yaw{self.vyaw:.1f}"
        if self.family.startswith("mat"):
            top = "soft" if self.soft else "rigid"
            return f"{self.family}/{top}/{round(self.height_m * 1000):d}mm/vx{self.vx:.2f}"
        return f"{self.family}/{self.entry}"


def cells() -> list[Cell]:
    out = [
        Cell("turn", "stand", 0.0, 1.0),
        Cell("turn", "walk", 0.0, 1.0),
        Cell("turn", "walk", 0.20, 1.0),
        Cell("turn", "walk", 0.30, 1.0),
        Cell("turn", "stand", 0.35, 1.0),
        Cell("march", "stand", 0.20, 0.0),
        Cell("march", "walk", 0.20, 0.0),
        Cell("bow", "stand"),
        Cell("lie", "stand"),
    ]
    out += [Cell("mat_up", "walk", 0.40, 0.0, h) for h in (0.03, 0.05)]
    out += [Cell("mat_dn", "stand", 0.40, 0.0, h) for h in (0.03, 0.04, 0.05)]
    out += [
        Cell("mat_up", "walk", 0.40, 0.0, 0.04, True),
        Cell("mat_dn", "stand", 0.40, 0.0, 0.04, True),
    ]
    out += [Cell("turn", "nudge", 0.0, 1.0, pre_s=t) for t in (0.6, 1.2)]
    return out


def trained_cells() -> list[Cell]:
    """The same measurements with the trained B entries (the after of every cell above)."""
    out = [Cell("turn", e, skill=k) for k in ("turn_left", "turn_right") for e in ("stand", "walk")]
    out += [Cell("march", e, skill="march") for e in ("stand", "walk")]
    out += [Cell("bow", "stand", skill="bow"), Cell("lie", "stand", skill="lie_low")]
    out += [Cell("mat_up", "stand", 0.0, 0.0, h, skill="step_up") for h in (0.03, 0.04, 0.05)]
    out += [Cell("mat_dn", "stand", 0.0, 0.0, h, skill="step_down") for h in (0.03, 0.04, 0.05)]
    out += [
        Cell("mat_up", "stand", 0.0, 0.0, 0.04, True, skill="step_up"),
        Cell("mat_dn", "stand", 0.0, 0.0, 0.04, True, skill="step_down"),
    ]
    return out


SKILL_HOLD_S = {"turn_left": TURN_S, "turn_right": TURN_S, "march": TURN_S, "lie_low": 3.0}
RE_REQUEST_S = 0.1
FIXED_DELAY: int | None = None  # --delay-steps: a fixed servo command delay on both servo models


def b_skilldefs(onnx_dir: str | Path, policies: Path) -> list[Any]:
    """The B table (bsk_contract, pinned scale/gain) with this folder's ONNX files, plus skill_abort
    (v5 velstand as a one-tick chaining skill: the neural duck's release of a held stepping skill)."""
    from duckmind.skilltrain import bsk_contract as B
    from duckmind.skilltrain.table import SkillDef

    folder = Path(onnx_dir)
    paths = {
        n: str((folder / f"{n}.onnx").resolve()) for n in B.NETS if (folder / f"{n}.onnx").is_file()
    }
    defs = [SkillDef(**d) for d in B.skill_defs(paths, nets=tuple(paths))]
    defs.append(SkillDef("skill_abort", str(policies / "velstand.onnx"), 0.02, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                         0.0, True, 0.9, 1.0))  # fmt: skip
    return defs


def seed_of(key: str, rep: int) -> int:
    return zlib.crc32(f"bsk-a/{key}/{rep}".encode()) & 0x7FFFFFFF


# ── robotd's ground pick on the Stage-1 port ────────────────────────────────────────────────


def _ground_pick_port(walk: Path, sitstand: Path, ground_pick: Path, head_alpha: float) -> Any:
    """A :class:`~duckmind.skilltrain.robotd_port.Robotd` that also runs robotd's ground pick
    (``control.rs`` ``start_ground_pick`` / ``step``; no configured skills)."""
    import numpy as np

    from duckmind.skilltrain import robotd_port as RP

    class GPPolicySet(RP.PolicySet):
        def __init__(self) -> None:
            super().__init__(walk, sitstand, [])
            self.ground_pick = RP.OnnxNet(ground_pick)

        def net(self, key: tuple[str, int]) -> Any:
            return self.ground_pick if key[0] == "ground_pick" else super().net(key)

    class GPController(RP.Controller):
        def __init__(self, policy: Any) -> None:
            super().__init__(policy, [])
            self.gp: float | None = None

        def start_ground_pick(self) -> tuple[bool, str]:
            if self.gp is not None:
                return False, "ground pick already running"
            self.gp = 0.0
            return True, "started"

        def busy(self) -> bool:
            return super().busy() or self.gp is not None

        def step(self, sensors: Any, command: Any, body_active: bool = False, dt: float = RP.DT,
                 scale_mult: float = 1.0, now: float | None = None) -> Any:  # fmt: skip
            if self.gp is None or self.active is not None:
                out = super().step(sensors, command, body_active, dt, scale_mult, now)
                if self.gp is not None:
                    self._advance(dt)
                return out
            if self.sit == "rising" and self.rise_remaining <= 0.0:
                self.sit = "up"
            angle = 2.0 * math.pi * self.gp
            effective = RP.Command(twist=(math.cos(angle), math.sin(angle), 0.0))
            net = ("ground_pick", 0)
            self.last_net = net
            obs = np.concatenate(
                [
                    np.asarray(sensors.gyro, float),
                    np.asarray(sensors.gravity, float),
                    np.asarray(sensors.q, float) - RP.HOME,
                    np.asarray(sensors.qd, float),
                    self.last_action.astype(float),
                    np.asarray(effective.twist, float),
                    np.zeros(4),
                    np.zeros(6),
                ]
            ).astype(np.float32)
            action = self.policy.infer(obs, net)
            self.last_action = action
            scale = 1.0 * scale_mult  # ground_pick_action_scale 1.0
            gain = int(round(self.tuning.gain * 1.0))  # ground_pick_gain_ratio 1.0
            targets = RP.HOME + scale * action.astype(float)
            t = self.tuning
            if self.previous is not None:
                hm = RP.HEAD_MASK
                if t.head_lowpass is not None:
                    targets[hm] = (
                        t.head_lowpass * targets[hm] + (1 - t.head_lowpass) * self.previous[hm]
                    )
                if t.legs_lowpass is not None:
                    targets[~hm] = (
                        t.legs_lowpass * targets[~hm] + (1 - t.legs_lowpass) * self.previous[~hm]
                    )
            self.previous = targets.copy()
            self._advance(dt)
            if self.sit == "rising":
                self.rise_remaining -= dt
            return RP.Step(targets, "ground_pick", gain, self.busy(), net, scale, effective)

        def _advance(self, dt: float) -> None:
            assert self.gp is not None
            self.gp += dt / GP_PERIOD_S
            if self.gp >= GP_END_PHASE:
                self.gp = None
                self.events.append({"t": None, "kind": "ended", "skill": "ground_pick"})

    class GPRobotd(RP.Robotd):
        def __init__(self) -> None:
            policy = GPPolicySet()
            super().__init__(policy, [], head_alpha=head_alpha)
            self.controller = GPController(policy)
            self._gp_request = False

        def request(self, name: str) -> tuple[bool, str]:
            if name == "ground_pick":
                if not self.ready:
                    return False, "the policy is not driving"
                self._gp_request = True
                return True, "queued"
            return super().request(name)

        def tick(self, sensors: Any, now: float, battery_v: float = RP.NOMINAL_V) -> Any:
            if self._gp_request and self.ready:
                ok, why = self.controller.start_ground_pick()
                self.controller.events.append({"t": round(now, 4), "kind": "started" if ok else "refused",
                                               "skill": "ground_pick", "why": why})  # fmt: skip
            self._gp_request = False
            return super().tick(sensors, now, battery_v)

    return GPRobotd()


# ── the duck ─────────────────────────────────────────────────────────────────────────────────

_DUCK: Any = None


def _duck_class() -> Any:
    global _DUCK
    if _DUCK is not None:
        return _DUCK
    from duckmind.skilltrain.skillduck import SkillDuck

    class BskDuck(SkillDuck):
        """The gate duck, also recording per tick: the beak tip's world height, both feet's world
        x/z (collision geom centres), trunk yaw and pitch (+ = nose down)."""

        def tick(self) -> None:
            super().tick()
            d = self.data
            ex = self.__dict__.setdefault("extra", [])
            q = [float(v) for v in d.qpos[self.qadr + 3 : self.qadr + 7]]
            w, x, y, z = q
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            fl, fr = (int(g) for g in self.foot_geoms)
            beak = float(d.site_xpos[self.beak_site][2]) if self.beak_site >= 0 else float("nan")
            ex.append((float(d.time), beak, float(d.geom_xpos[fl][0]), float(d.geom_xpos[fr][0]),
                       float(d.geom_xpos[fl][2]), float(d.geom_xpos[fr][2]), yaw, pitch))  # fmt: skip

    _DUCK = BskDuck
    return BskDuck


def _mat(cell: Cell) -> Any:
    """The mat as the hop probe's strip (0.40 m deep), with an optional compliant top."""
    from duckmind.skilltrain.hop_probe import Obstacle

    base = Obstacle(cell.height_m, "sharp", depth_m=MAT_DEPTH_M, width_m=2.0, face_x=MAT_FACE_X)
    if not cell.soft:
        return base

    @dataclass(frozen=True)
    class SoftMat(Obstacle):
        @property
        def name(self) -> str:
            return "soft-mat-" + super().name

        def xml(self) -> str:
            sr = " ".join(f"{v:g}" for v in SOFT_SOLREF)
            si = " ".join(f"{v:g}" for v in SOFT_SOLIMP)
            fr = " ".join(f"{v:g}" for v in SOFT_FRICTION)
            return (
                super()
                .xml()
                .replace(
                    "<geom ", f'<geom priority="2" friction="{fr}" solref="{sr}" solimp="{si}" '
                )
            )

    return SoftMat(cell.height_m, "sharp", depth_m=MAT_DEPTH_M, width_m=2.0, face_x=MAT_FACE_X)


def _mat_name(cell: Cell) -> str:
    return ("soft-" if cell.soft else "") + f"mat-{round(cell.height_m * 1000):d}mm"


# ── one trial ────────────────────────────────────────────────────────────────────────────────


def _run(duck: Any, seconds: float, each: Any = None) -> None:
    for _ in range(max(0, int(round(seconds * CONTROL_HZ)))):
        if each is not None:
            each(duck.now())
        duck.tick()


def _yaw_of(duck: Any) -> float:
    w, x, y, z = (float(v) for v in duck.data.qpos[duck.qadr + 3 : duck.qadr + 7])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_trial(cell: Cell, *, servo: str, rep: int, scenes: Path, policies: str | None = None,
              onnx_dir: str | None = None) -> dict[str, Any]:  # fmt: skip
    import numpy as np

    from duckmind.skilltrain import dr as DR
    from duckmind.world.headless_duck import official_dir, policy_dir

    began = time.perf_counter()
    seed = seed_of(cell.key, rep)
    draw = DR.draw(seed, "recipe" if servo == "bam" else "physical")
    rng = np.random.default_rng(seed + 11)
    folder = None
    if cell.family.startswith("mat"):
        from duckmind.skilltrain.hop_probe import scene_dir

        folder = scene_dir(_mat(cell), scenes, official_dir(None))
    Duck = _duck_class()
    pol0 = Path(policies) if policies else policy_dir()
    skills = b_skilldefs(onnx_dir, pol0) if cell.skill else []
    if cell.skill and cell.skill not in {d.name for d in skills}:
        raise ValueError(f"{cell.skill}: no ONNX for it in {onnx_dir}")
    duck = Duck(skills, servo=servo, draw=draw, head_alpha=1.0, keyframe="SIT", policies=policies,
                mjcf_dir=folder, x=0.0 if folder is None else MAT_FACE_X - 0.9,
                y=0.0, yaw=float(rng.uniform(-math.pi, math.pi)))  # fmt: skip
    pol = pol0
    if FIXED_DELAY is not None:
        from collections import deque

        duck.lag_steps = int(FIXED_DELAY)
        duck._lag = deque(maxlen=duck.lag_steps + 1)
    if cell.family == "bow" and not cell.skill:
        duck.port = _ground_pick_port(pol / "velstand.onnx", pol / "alpha_sitstand.onnx",
                                      pol / "alpha_ground_pick.onnx", 1.0)  # fmt: skip
        duck.skill_log = duck.port.controller.events
    row: dict[str, Any] = {"key": f"{cell.key}/{servo}/{rep}", "cell": asdict(cell), "servo": servo, "rep": rep,
                           "seed": seed, "dr": draw.to_json(), "excluded": None}  # fmt: skip

    def excluded(why: str) -> dict[str, Any]:
        row["excluded"] = why
        row["wall_s"] = round(time.perf_counter() - began, 3)
        return row

    ok, _ = duck.stand(timeout=10.0, hold=1.0)
    if not ok:
        return excluded("did not stand up after the seated boot")
    ex0 = np.asarray(duck.extra[-10:], float)
    duck._rest_z = float(np.min(np.minimum(ex0[:, 4], ex0[:, 5])))  # a sole on the floor, standing
    row["delay_steps"] = int(duck.lag_steps)
    fam = cell.family
    if cell.skill:
        row.update(_skill_trial(duck, cell, rng))
        if row.get("excluded"):
            return excluded(row["excluded"])
        row["sim_s"] = round(duck.now(), 3)
        row["wall_s"] = round(time.perf_counter() - began, 3)
        return row
    if fam in ("turn", "march"):
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        if cell.entry == "walk":
            _run(duck, float(rng.uniform(3.0, 3.5)), lambda _t: duck.move(0.40))
        elif cell.entry == "nudge":
            _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
            _run(duck, cell.pre_s, lambda _t: duck.move(0.40))
        else:
            _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        if duck.safety.fallen or duck.tilt_deg() > 30.0:
            return excluded("down before the command")
        t0 = duck.now()
        _run(duck, TURN_S, lambda _t: duck.move(cell.vx, 0.0, sign * cell.vyaw))
        t1 = duck.now()
        _run(duck, AFTER_S, lambda _t: duck.move(0.0))
        row.update(gait_metrics(duck, t0, t1, sign))
        row["sign"] = sign
    elif fam == "bow":
        _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        t0 = duck.now()
        duck.do("ground_pick")
        t_end = None
        for _ in range(int(4.0 * CONTROL_HZ)):
            duck.move(0.0)
            duck.tick()
            if duck.policy != "ground_pick" and duck.now() - t0 > 0.1:
                t_end = duck.now()
                break
        if t_end is None:
            return excluded("the ground pick did not hand back within 4 s")
        _run(duck, HANDBACK_S, lambda _t: duck.move(0.0))
        row.update(bow_metrics(duck, t0, t_end))
    elif fam == "lie":
        _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        t_sit = duck.now()
        duck.do("sit_toggle")
        waited = 0.0
        while duck.policy != "sit" and waited < 6.0:
            duck.tick()
            waited += DT
        if duck.policy != "sit":
            return excluded("did not sit down")
        _run(duck, 2.5)
        t_head = duck.now()
        _run(duck, 2.0, lambda _t: duck.head(LIE_HEAD))
        t_head_end = duck.now()
        duck.head((0.0, 0.0, 0.0, 0.0))
        _run(duck, 0.8)
        t_rise = duck.now()
        duck.do("sit_toggle")
        _run(duck, 4.0, lambda _t: duck.move(0.0))
        row.update(lie_metrics(duck, t_sit, t_head, t_head_end, t_rise))
    elif fam in ("mat_up", "mat_dn"):
        row.update(_mat_trial(duck, cell, rng))
        if row.get("excluded"):
            return excluded(row["excluded"])
    else:
        raise ValueError(f"family {fam!r}")
    row["sim_s"] = round(duck.now(), 3)
    row["wall_s"] = round(time.perf_counter() - began, 3)
    return row


def _held(duck: Any, name: str, seconds: float) -> None:
    """Hold a chaining entry as the neural duck does: re-request every 0.1 s, zero twist."""
    last = -1e9
    for _ in range(int(round(seconds * CONTROL_HZ))):
        if duck.now() - last >= RE_REQUEST_S - 1e-9:
            duck.do(name)
            last = duck.now()
        duck.move(0.0)
        duck.tick()


def _until_label_leaves(duck: Any, names: tuple[str, ...], limit_s: float) -> float | None:
    started = False
    for _ in range(int(limit_s * CONTROL_HZ)):
        duck.move(0.0)
        duck.tick()
        mine = any(duck.policy == n or duck.policy == f"{n}:unwind" for n in names)
        started = started or mine
        if started and not mine:
            return duck.now()
    return None


def _skill_trial(duck: Any, cell: Cell, rng: Any) -> dict[str, Any]:
    """One trained B entry, as the neural duck would ask for it."""
    fam, name = cell.family, str(cell.skill)
    if fam in ("turn", "march"):
        if cell.entry == "walk":
            _run(duck, float(rng.uniform(3.0, 3.5)), lambda _t: duck.move(0.40))
        else:
            _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        if duck.safety.fallen or duck.tilt_deg() > 30.0:
            return {"excluded": "down before the skill"}
        t0 = duck.now()
        _held(duck, name, SKILL_HOLD_S[name])
        t1 = duck.now()
        duck.do("skill_abort")
        _run(duck, AFTER_S, lambda _t: duck.move(0.0))
        sign = -1.0 if name == "turn_right" else 1.0
        return {
            **gait_metrics(duck, t0, t1, sign),
            "sign": sign,
            "labels_hold": _labels(duck, t0, t1),
        }
    if fam in ("bow", "lie"):
        _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
        t0 = duck.now()
        if name == "lie_low":
            _held(duck, name, SKILL_HOLD_S[name])
            duck.do("bow_release")
            t_end = _until_label_leaves(duck, ("bow_release",), 3.0)
        else:
            duck.do(name)
            t_end = _until_label_leaves(duck, (name,), 4.0)
        if t_end is None:
            return {"excluded": f"{name} did not hand back"}
        _run(duck, HANDBACK_S, lambda _t: duck.move(0.0))
        return bow_metrics(duck, t0, t_end)
    # mats: standing 7–14 cm (trunk centre) before the edge, facing it ±8° (the training entries)
    h = cell.height_m
    aim = math.radians(float(rng.uniform(-8.0, 8.0)))
    y0 = float(rng.uniform(-0.3, 0.3))
    d = float(rng.uniform(0.07, 0.14))
    edge_x = MAT_FACE_X + (MAT_DEPTH_M if fam == "mat_dn" else 0.0)
    duck.teleport(edge_x - d, y0, aim)
    _run(duck, float(rng.uniform(0.5, 1.0)), lambda _t: duck.move(0.0))
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return {"excluded": "down before the skill"}
    t0 = duck.now()
    duck.do(name)
    t_end = _until_label_leaves(duck, (name,), 5.0)
    _run(duck, 3.0, lambda _t: duck.move(0.0))
    import numpy as np

    ex = np.asarray(duck.extra[-5:], float)
    xl, xr, zl, zr = ex[-1, 2], ex[-1, 3], ex[-1, 4], ex[-1, 5]
    rest = _foot_rest_z(duck)
    past = min(xl, xr) > edge_x + PAST_FOOT_M * 0.5
    if fam == "mat_up":
        over = past and min(zl, zr) > rest + 0.5 * h
    else:
        over = past and max(zl, zr) < rest + 0.5 * h
    over = bool(over and duck.tilt_deg() < 45.0 and not duck.safety.fallen)
    res = mat_metrics(duck, cell, t0, t0, t_end if over else None, over, None)
    res["handed_back"] = t_end is not None
    return res


def _labels(duck: Any, t0: float, t1: float) -> dict[str, int]:
    import numpy as np

    tr = duck.trace.arrays()
    sel = (tr["t"] > t0) & (tr["t"] <= t1)
    labs, counts = np.unique(tr["label"][sel], return_counts=True)
    return {str(a): int(b) for a, b in zip(labs, counts, strict=True)}


def _mat_trial(duck: Any, cell: Cell, rng: Any) -> dict[str, Any]:
    """Walk at the mat edge (up) or off it (down) with the heading hold; classify the outcome."""
    import numpy as np

    h = cell.height_m
    aim = math.radians(float(rng.uniform(-10.0, 10.0)))
    y0 = float(rng.uniform(-0.3, 0.3))
    if cell.family == "mat_up":
        d0 = float(rng.uniform(0.30, 0.45))
        duck.teleport(MAT_FACE_X - d0, y0, aim)
        edge_x = MAT_FACE_X
    else:
        back = float(rng.uniform(0.15, 0.25))
        duck.teleport(MAT_FACE_X + MAT_DEPTH_M - back, y0, aim)
        edge_x = MAT_FACE_X + MAT_DEPTH_M
    start_i = len(duck.extra)
    _run(duck, float(rng.uniform(0.5, 1.0)), lambda _t: duck.move(0.0))
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return {"excluded": "down before the walk"}
    sink_mm = None
    if cell.family == "mat_dn":
        ex = np.asarray(duck.extra[start_i:], float)
        sink_mm = round(
            1000
            * (h - float(np.median(np.minimum(ex[-10:, 4], ex[-10:, 5])) - _foot_rest_z(duck))),
            2,
        )
    t_walk = duck.now()
    t_touch = t_done = t_stop = None
    over = False
    limit = 10.0
    while True:
        now = duck.now()
        walking = t_stop is None
        if walking:
            err = _wrap(_yaw_of(duck) - aim)
            duck.move(cell.vx, 0.0, max(-HOLD_MAX, min(HOLD_MAX, -HOLD_K * err)))
        else:
            duck.move(0.0)
        duck.tick()
        now = duck.now()
        _, _, xl, xr, zl, zr, _, _ = duck.extra[-1]
        trunk_x = float(duck.data.qpos[duck.qadr])
        lead = max(xl, xr)
        if t_touch is None and lead + 0.027 >= edge_x:
            t_touch = now  # the leading sole's toe reaches the edge line
        if cell.family == "mat_up":
            done = (min(xl, xr) > edge_x + PAST_FOOT_M and trunk_x > edge_x + PAST_TRUNK_M
                    and min(zl, zr) > h * 0.5 + _foot_rest_z(duck))  # fmt: skip
        else:
            done = (min(xl, xr) > edge_x + PAST_FOOT_M and trunk_x > edge_x + PAST_TRUNK_M
                    and max(zl, zr) < 0.5 * h + _foot_rest_z(duck))  # fmt: skip
        if walking:
            if done and t_done is None and duck.tilt_deg() < 45.0 and not duck.safety.fallen:
                t_done = now
                over = True
            if duck.safety.fallen:
                t_stop = now + 3.0
            elif t_done is not None and now >= t_done + 1.0:
                t_stop = now + 1.5
            elif t_touch is not None and now >= t_touch + 4.0:
                t_stop = now + 1.5
            elif now - t_walk > limit:
                t_stop = now + 1.0
        elif now >= t_stop:
            break
    return mat_metrics(duck, cell, t_walk, t_touch, t_done, over, sink_mm)


def _foot_rest_z(duck: Any) -> float:
    """The foot geom centre's height with the sole on the floor (the standing duck on the official
    floor, recorded by :func:`run_trial` right after it stood up, before any teleport)."""
    return float(duck.__dict__["_rest_z"])


# ── metrics (pure numpy on the recorded arrays) ──────────────────────────────────────────────


def _arrays(duck: Any) -> tuple[dict[str, Any], Any]:
    import numpy as np

    tr = duck.trace.arrays()
    ex = np.asarray(duck.extra, float)
    n = min(len(tr["t"]), len(ex))
    tr = {k: v[:n] for k, v in tr.items()}
    return tr, ex[:n]


def _strain(tr: dict[str, Any], sel: Any) -> dict[str, Any]:
    import numpy as np

    tau = np.abs(tr["tau_pk"][sel])
    qd = np.abs(tr["qd"][sel])
    legs, head = list(LEGS), list(HEAD)
    if not len(tau):
        return {}
    stall = (tau > TORQUE_NM) & (qd < STALL_VEL)
    upright = tr["tilt"][sel] < 45.0
    hous = tr["f_housing"][sel]
    impacts = int(
        ((hous.max(axis=1) if hous.size else np.zeros(len(tau))) > HOUSING_N)[upright].sum()
    )
    return {
        "leg_tau_p95": round(float(np.percentile(tau[:, legs], 95)), 4),
        "leg_tau_max": round(float(tau[:, legs].max()), 4),
        "head_tau_p95": round(float(np.percentile(tau[:, head], 95)), 4),
        "over04_share": round(float((tau[:, legs].max(axis=1) > TORQUE_NM).mean()), 4),
        "stall_share": round(float(stall.any(axis=1).mean()), 4),
        "housing_contact_ticks": impacts,
    }


def _fell(tr: dict[str, Any], t0: float) -> tuple[bool, float]:
    after = tr["t"] >= t0
    fallen = (tr["fallen"] > 0.5) & after
    return bool(fallen.any()), float(tr["tilt"][after].max()) if after.any() else float("nan")


def gait_metrics(duck: Any, t0: float, t1: float, sign: float) -> dict[str, Any]:
    import numpy as np

    tr, ex = _arrays(duck)
    t = tr["t"]
    win = (t > t0) & (t <= t1 + 1e-9)
    idx = np.flatnonzero(win)
    yaw = np.unwrap(ex[:, 6])
    i0, i1 = idx[0] - 1, idx[-1]
    im = int(np.searchsorted(t, t1 - 2.0))
    dyaw = sign * math.degrees(yaw[i1] - yaw[i0])
    rate2 = sign * math.degrees(yaw[i1] - yaw[im]) / 2.0
    xy = tr["pos"][:, :2]
    seg = xy[i0 : i1 + 1]
    path = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
    net = float(np.linalg.norm(seg[-1] - seg[0]))
    excursion = float(np.linalg.norm(seg - seg[0], axis=1).max())
    td = tr["td"][win].sum(axis=1)
    last2 = (t > t1 - 2.0) & (t <= t1 + 1e-9)
    z0 = ex[i0, 4:6]
    lift = float((ex[win, 4:6] - z0).max()) if win.any() else 0.0
    fell, max_tilt = _fell(tr, t0)
    dy = abs(math.radians(dyaw))
    return {
        "dyaw_deg": round(dyaw, 2),
        "rate_last2_dps": round(rate2, 2),
        "path_m": round(path, 4),
        "net_m": round(net, 4),
        "excursion_m": round(excursion, 4),
        "radius_m": round(path / dy, 4) if dy > math.radians(5) else None,
        "footfalls": int(td.sum()),
        "footfalls_last2": int(tr["td"][last2].sum()),
        "foot_lift_mm": round(1000 * lift, 2),
        "fell": fell,
        "max_tilt_deg": round(max_tilt, 2),
        "speed_last2_mps": round(float(np.linalg.norm(xy[i1] - xy[im]) / 2.0), 4),
        **_strain(tr, win),
    }


def bow_metrics(duck: Any, t0: float, t_end: float) -> dict[str, Any]:
    import numpy as np

    tr, ex = _arrays(duck)
    t = tr["t"]
    win = (t > t0) & (t <= t_end + 1e-9)
    pre = np.flatnonzero(t <= t0)[-5:]
    beak0 = float(np.median(ex[pre, 1]))
    z0 = float(np.median(tr["pos"][pre, 2]))
    pitch0 = float(np.median(ex[pre, 7]))
    beak = ex[win, 1]
    i_low = int(np.argmin(beak))
    low = float(beak[i_low])
    t_low = float(t[win][i_low] - t0)
    near = beak <= low + 0.010
    hold = float(near.sum() * DT)
    xy = tr["pos"][:, :2]
    iw = np.flatnonzero(win)
    drift = float(np.linalg.norm(xy[iw[-1]] - xy[iw[0] - 1]))
    fell_in = bool((tr["fallen"][win] > 0.5).any())
    after = t > t_end
    fell_after = bool((tr["fallen"][after] > 0.5).any())
    head_touch = int((tr["f_head"][win] > 1.0).sum())
    end_i = iw[-1]
    back = t > t_end + HANDBACK_S - 0.5
    # b:train: the gate's poses over time (bsk_contract.GATES) — bow: pitch ≥ 15° with the beak ≥ 40 mm lower;
    # lie: pitch ≥ 20° with the trunk ≥ 20 mm lower (first time after the request, and ticks held in the window)
    pitch_w = ex[win, 7] - pitch0
    drop_w = z0 - tr["pos"][win, 2]
    bow_pose = (pitch_w >= 15.0) & ((beak0 - beak) >= 0.040)
    lie_pose = (pitch_w >= 20.0) & (drop_w >= 0.020)
    tw = t[win] - t0

    def first(mask: Any) -> float | None:
        return round(float(tw[np.flatnonzero(mask)[0]]), 3) if mask.any() else None

    return {
        "t_bow_s": first(bow_pose),
        "hold_bow_s": round(float(bow_pose.sum() * DT), 3),
        "t_lie_s": first(lie_pose),
        "hold_lie_s": round(float(lie_pose.sum() * DT), 3),
        "beak_start_mm": round(1000 * beak0, 1),
        "beak_min_mm": round(1000 * low, 1),
        "beak_drop_mm": round(1000 * (beak0 - low), 1),
        "t_lowest_s": round(t_low, 3),
        "low_hold_s": round(hold, 3),
        "pitch_down_max_deg": round(float(ex[win, 7].max() - pitch0), 2),
        "trunk_dip_mm": round(1000 * (z0 - float(tr["pos"][win, 2].min())), 1),
        "drift_m": round(drift, 4),
        "footfalls": int(tr["td"][win].sum()),
        "head_floor_ticks": head_touch,
        "fell_in": fell_in,
        "fell_after": fell_after,
        "handback_trunk_dz_mm": round(1000 * (float(tr["pos"][end_i, 2]) - z0), 1),
        "handback_pitch_deg": round(float(ex[end_i, 7] - pitch0), 2),
        "handback_beak_mm": round(1000 * float(ex[end_i, 1]), 1),
        "settled_dz_mm": round(1000 * (float(np.median(tr["pos"][back, 2])) - z0), 1)
        if back.any()
        else None,
        "duration_s": round(t_end - t0, 3),
        **_strain(tr, win),
    }


def lie_metrics(
    duck: Any, t_sit: float, t_head: float, t_head_end: float, t_rise: float
) -> dict[str, Any]:
    import numpy as np

    tr, ex = _arrays(duck)
    t = tr["t"]
    pre = np.flatnonzero(t <= t_sit)[-5:]
    z0 = float(np.median(tr["pos"][pre, 2]))
    beak0 = float(np.median(ex[pre, 1]))
    pitch0 = float(np.median(ex[pre, 7]))
    seated = (t > t_head - 1.0) & (t <= t_head)
    look = (t > t_head) & (t <= t_head_end)
    after = t > t_rise
    labels = tr["label"]
    risen = [
        i for i in np.flatnonzero(after) if labels[i] in ("walk", "stand") and tr["tilt"][i] < 25.0
    ]
    t_up = float(t[risen[0]] - t_rise) if risen else None
    return {
        "seated_trunk_z_mm": round(1000 * float(np.median(tr["pos"][seated, 2])), 1),
        "seated_drop_mm": round(1000 * (z0 - float(np.median(tr["pos"][seated, 2]))), 1),
        "seated_pitch_deg": round(float(np.median(ex[seated, 7]) - pitch0), 2),
        "beak_standing_mm": round(1000 * beak0, 1),
        "beak_seated_mm": round(1000 * float(np.median(ex[seated, 1])), 1),
        "beak_head_down_min_mm": round(1000 * float(ex[look, 1].min()), 1),
        "pitch_head_down_deg": round(float(ex[look, 7].max() - pitch0), 2),
        "fell_sit": bool((tr["fallen"][(t > t_sit) & (t <= t_rise)] > 0.5).any()),
        "fell_rise": bool((tr["fallen"][after] > 0.5).any()),
        "rise_s": None if t_up is None else round(t_up, 3),
        **_strain(tr, (t > t_sit) & (t <= t_rise)),
    }


def mat_metrics(duck: Any, cell: Cell, t_walk: float, t_touch: float | None, t_done: float | None,
                over: bool, sink_mm: float | None) -> dict[str, Any]:  # fmt: skip
    import numpy as np

    tr, _ = _arrays(duck)
    t = tr["t"]
    t0 = (t_touch if t_touch is not None else t_walk) - 0.5
    after = t >= t0
    fallen = (tr["fallen"] > 0.5) & after
    fell = bool(fallen.any())
    run = 0
    stuck = False
    for i in np.flatnonzero(after):
        run = run + 1 if (tr["tilt"][i] > STUCK_DEG and tr["fallen"][i] < 0.5) else 0
        if run >= int(STUCK_S * CONTROL_HZ):
            stuck = True
            break
    max_tilt = float(tr["tilt"][after].max()) if after.any() else float("nan")
    if fell:
        outcome = "fall"
    elif stuck and not over:
        outcome = "stuck"
    elif over:
        outcome = "done"
    else:
        outcome = "blocked"
    return {
        "outcome": outcome,
        "stumble": bool(max_tilt >= STUMBLE_DEG),
        "clean": bool(outcome == "done" and max_tilt < STUMBLE_DEG),
        "max_tilt_deg": round(max_tilt, 2),
        "t_done_after_touch": None
        if (t_done is None or t_touch is None)
        else round(t_done - t_touch, 3),
        "touched": t_touch is not None,
        "sink_mm": sink_mm,
        **_strain(tr, after),
    }


# ── runner / summary ─────────────────────────────────────────────────────────────────────────


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
    scenes = out / "scenes"
    path = out / f"rows-{args.servo}.jsonl"
    done = _done(path)
    guard = Guard(not args.no_guard, log=lambda s: print(s, flush=True))
    global FIXED_DELAY
    FIXED_DELAY = None if args.delay_steps is None else int(args.delay_steps)
    pool = trained_cells() if args.onnx_dir else cells()
    todo = [
        c for c in pool if not args.only or any(c.key.startswith(p) for p in args.only.split(","))
    ]
    began = time.perf_counter()
    count = 0
    with path.open("a") as f:
        for cell in todo:
            valid = sum(1 for r in done.values() if r["cell"] == asdict(cell) and r["servo"] == args.servo
                        and not r["excluded"])  # fmt: skip
            rep = 0
            while valid < args.n and rep < 3 * args.n:
                key = f"{cell.key}/{args.servo}/{rep}"
                if key in done:
                    rep += 1
                    continue
                guard.wait()
                try:
                    row = run_trial(
                        cell, servo=args.servo, rep=rep, scenes=scenes, onnx_dir=args.onnx_dir
                    )
                except Exception as exc:  # noqa: BLE001 - record and go on
                    row = {"key": key, "cell": asdict(cell), "servo": args.servo, "rep": rep,
                           "excluded": f"error: {type(exc).__name__}: {exc}"}  # fmt: skip
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[key] = row
                count += 1
                if not row["excluded"]:
                    valid += 1
                rep += 1
            print(f"[{args.servo}] {cell.key}: {valid} valid; {count} trials, {time.perf_counter() - began:.0f} s",
                  flush=True)  # fmt: skip
    meta = {"servo": args.servo, "n": args.n, "trials_this_run": count,
            "wall_s": round(time.perf_counter() - began, 1), "guard_pauses": guard.pauses,
            "realtime_min": min(guard.readings, default=None)}  # fmt: skip
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


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f}%]"


def _q(vals: list[float]) -> str:
    import numpy as np

    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "—"
    a = np.asarray(v, float)
    return f"{np.median(a):.3g} ({np.percentile(a, 10):.3g}–{np.percentile(a, 90):.3g})"


SUMMARY_FIELDS = {
    "turn": ("dyaw_deg", "rate_last2_dps", "excursion_m", "radius_m", "footfalls", "footfalls_last2",
             "foot_lift_mm", "speed_last2_mps", "leg_tau_p95", "max_tilt_deg"),
    "march": ("footfalls", "footfalls_last2", "net_m", "speed_last2_mps", "foot_lift_mm", "dyaw_deg",
              "leg_tau_p95", "max_tilt_deg"),
    "bow": ("t_bow_s", "hold_bow_s", "t_lie_s", "hold_lie_s",
            "beak_start_mm", "beak_min_mm", "beak_drop_mm", "t_lowest_s", "low_hold_s", "pitch_down_max_deg",
            "trunk_dip_mm", "drift_m", "footfalls", "handback_trunk_dz_mm", "handback_pitch_deg", "settled_dz_mm",
            "leg_tau_p95", "head_tau_p95", "head_floor_ticks"),
    "lie": ("seated_trunk_z_mm", "seated_drop_mm", "seated_pitch_deg", "beak_standing_mm", "beak_seated_mm",
            "beak_head_down_min_mm", "pitch_head_down_deg", "rise_s", "leg_tau_p95"),
    "mat_up": ("t_done_after_touch", "max_tilt_deg", "leg_tau_p95", "leg_tau_max"),
    "mat_dn": ("t_done_after_touch", "max_tilt_deg", "sink_mm", "leg_tau_p95", "leg_tau_max"),
}  # fmt: skip


def summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    rows: list[dict[str, Any]] = []
    for servo in ("mjcf", "bam"):
        rows += list(_done(out / f"rows-{servo}.jsonl").values())
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    excluded: dict[str, int] = {}
    for r in rows:
        if r["excluded"]:
            excluded[r["excluded"][:60]] = excluded.get(r["excluded"][:60], 0) + 1
            continue
        groups.setdefault((Cell(**r["cell"]).key, r["servo"]), []).append(r)
    lines = ["# B-list skills, part (a): v5 today (bsk_probe)", "",
             "p50 (p10–p90) over n DR ducks; rates k/n [Wilson 95 %].", ""]  # fmt: skip
    summary: dict[str, Any] = {"excluded": excluded, "cells": []}
    for (key, servo), rs in sorted(groups.items()):
        fam = rs[0]["cell"]["family"]
        if rs[0]["cell"].get("skill") and fam == "lie":
            fam = (
                "bow"  # a trained lie_low is scored like a bow (front-down depth, hold, hand-back)
            )
        n = len(rs)
        cell: dict[str, Any] = {"key": key, "servo": servo, "n": n}
        if fam in ("turn", "march"):
            cell["falls"] = rate(sum(r["fell"] for r in rs), n)
            cell["stepping_last2"] = rate(sum(r["footfalls_last2"] >= 2 for r in rs), n)
            cell["on_spot_45"] = rate(
                sum(r["dyaw_deg"] >= 45 and r["excursion_m"] <= 0.06 for r in rs), n
            )
            cell["turned_90"] = rate(sum(r["dyaw_deg"] >= 90 for r in rs), n)
        elif fam == "bow":
            cell["falls_in"] = rate(sum(r["fell_in"] for r in rs), n)
            cell["falls_handback"] = rate(sum(r["fell_after"] for r in rs), n)
            cell["pitch_ge_20"] = rate(sum(r["pitch_down_max_deg"] >= 20 for r in rs), n)
        elif fam == "lie":
            cell["falls_sit"] = rate(sum(r["fell_sit"] for r in rs), n)
            cell["falls_rise"] = rate(sum(r["fell_rise"] for r in rs), n)
            cell["risen_3s"] = rate(
                sum(r["rise_s"] is not None and r["rise_s"] <= 3.0 for r in rs), n
            )
        else:
            for o in ("done", "blocked", "stuck", "fall"):
                cell[o] = rate(sum(r["outcome"] == o for r in rs), n)
            cell["clean"] = rate(sum(r["clean"] for r in rs), n)
            cell["stumble"] = rate(sum(r["stumble"] for r in rs), n)
        for fld in SUMMARY_FIELDS.get(fam, ()):
            cell[fld] = _q([r.get(fld) for r in rs])
        summary["cells"].append(cell)
        lines.append(f"## {key} · {servo} · n = {n}")
        lines += [f"- {k}: {v}" for k, v in cell.items() if k not in ("key", "servo", "n")]
        lines.append("")
    if excluded:
        lines.append(
            "Excluded before the command (replaced): " + json.dumps(excluded, ensure_ascii=False)
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsk_probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--servo", choices=("mjcf", "bam"), required=True)
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--out", required=True)
    r.add_argument("--only", default="", help="comma-separated key prefixes (e.g. turn/,bow/)")
    r.add_argument("--no-guard", action="store_true")
    r.add_argument(
        "--onnx-dir",
        default=None,
        help="trained B networks (bskstep/bskbow/bskmat.onnx): the trained cells",
    )
    r.add_argument(
        "--delay-steps",
        type=int,
        default=None,
        help="fixed servo command delay, 5 ms steps, both models",
    )
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
