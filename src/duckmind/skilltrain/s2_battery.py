"""The Stage-2 battery on the official physics (tier 3 of the training plan §6, and the checkpoint
evaluations): a candidate walk network against v5, with the plan's gates.

    python -m duckmind.skilltrain.s2_battery --walk CAND.onnx|CKPT.pt --policies DIR --out OUT.json \
        [--mode full|quick|v5ctl] [--delays j3-6,d0,d20,d40] [--n 512] [--device cuda:0] [--stage pose|gaze]

**Why this harness.** It is the A100 neuro-validation run's (``neuro_a100``, 2026-09-28, 102 conditions,
53,248 DR ducks; its source is in the measurement records, not in the repo, and is bundled on Colab):
the official VelStand physics (``robot_allcollisions``, BAM XL330, the recipe's final DR), robotd's
command and action path re-implemented batched (twist/head/body EMAs, pose mode, the Sit machine,
scale, gain, voltage adaptation, low-pass, the fall verdict), the same programs and the same summaries —
so the candidate's regression numbers compare directly with v5's stored ones (``results.json``). Here
the walk slot runs the candidate (an ONNX or a training checkpoint's actor), the stand slot stays empty
(v5's ``stand = "none"``: pose mode drives the walk network), and the Stage-2 programs are added:

=====================  ======================================================================
``s2lean``             robot.pose ramped in at 0.5 rad/s / 60 mm/s (duck-body), held 4.5 s, ramped
                       out: achieved, t90, steps, the release rate, each foot's vertical load
``s2head``             standing neck pull-back (+0.3 / −0.2) → neck gain; ±3° yaw at 1 Hz → gain
``s2gaze``             walking 0.40 / 0.55 at head_alpha 1.0, head HOME / ±1.0 yaw / saccades: camera ⊥
                       RMS, the head camera's world-yaw SD in the stride band, delivered yaw, neck travel
``s2stopgo``           3 × (walk 3 s → stop → 0 / 0.7 / 1.5 s → pose sway 3 s → ramp release → walk)
=====================  ======================================================================

**Delays.** ``j3-6`` = the recipe's 15–30 ms per physics step (the stored v5 baselines'); ``dN`` = every
target exactly N ms late (the owner's simulator today is d0; WORK-RULES §0: the gates hold at 0/20/40).
``full`` runs the whole regression set at ``j3-6`` plus the core set at each fixed delay; ``quick``
(checkpoints) the core set at n ≈ 48; ``v5ctl`` the Stage-2 programs on v5 itself (the baselines the
stored run lacks), once per campaign.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.skilltrain import s2_contract as C

NEURO_RECORDS = paths.data_dir() / "a100" / "neuro-validation-2026-09-28"  # measurement records
NEURO_SRC_DEFAULT = NEURO_RECORDS / "prep" / "source"
V5_RESULTS_DEFAULT = NEURO_RECORDS / "results.json"
FORMAT = "duckmind-s2-battery-1"


def _neuro() -> Any:
    """Import the neuro_a100 package (bundled next to ours on Colab; the data folder on the Mac)."""
    src = os.environ.get("DUCKMIND_NEURO_A100_SRC") or str(NEURO_SRC_DEFAULT)
    if src not in sys.path:
        sys.path.insert(0, src)
    import neuro_a100  # noqa: F401
    from neuro_a100 import programs, report, robotd, runner, sim

    return {"P": programs, "R": robotd, "run": runner, "sim": sim, "rep": report}


# ── conditions ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cond:
    """A battery condition (duck-typed for neuro_a100's packer and batch runner)."""

    key: str
    block: str
    kind: str
    params: dict = field(default_factory=dict, hash=False, compare=False)
    n: int = 512
    stand_slot: bool = False
    head_alpha: float = 0.2
    group: str = ""
    seconds_: float = 0.0

    @property
    def seconds(self) -> float:
        if self.seconds_:
            return self.seconds_
        from neuro_a100 import conditions as NC

        return NC.duration_s(self.kind, self.params)

    @property
    def ticks(self) -> int:
        return int(round(self.seconds * 50))


def _std(
    key: str, kind: str, params: dict, block: str, head_alpha: float = 0.2, n: int = 512
) -> Cond:
    return Cond(key, block, kind, params, n=n, head_alpha=head_alpha)


def regression_conditions(n: int = 512) -> list[Cond]:
    """The stored battery's v5 conditions, with the candidate in the walk slot (no stand slot)."""
    _neuro()
    from neuro_a100 import conditions as NC

    out: list[Cond] = []
    for c in NC.all_conditions():
        if c.stand_slot:
            continue
        nn = 2 * n if c.kind == "posture" else n
        out.append(
            Cond(
                c.key,
                c.block,
                c.kind,
                c.params,
                n=nn,
                stand_slot=False,
                head_alpha=c.head_alpha,
                group=c.group,
            )
        )
    # the alpha_stand slot's lean / sway programs, now on the walk network itself
    leans = [("roll", s * a) for a in (0.1, 0.2) for s in (1, -1)]
    leans += [("pitch", s * a) for a in (0.1, 0.2) for s in (1, -1)]
    leans += [("z", z) for z in (-0.020, -0.010, 0.008)]
    for axis, value in leans:
        name = f"{axis}{'+' if value > 0 else '-'}{abs(value):g}"
        key = f"C.pose.walk.{name}"
        if any(c.key == f"C.pose.v5.{name}" for c in out):
            continue
        out.append(
            Cond(key, "C", "pose", {"pose": {"kind": "static", "axis": axis, axis: value}}, n=n)
        )
    for axis, amp, center, hz in (("roll", 0.2, 0.0, 0.25), ("roll", 0.1, 0.0, 0.5), ("pitch", 0.2, 0.0, 0.25),
                                  ("z", 0.014, -0.006, 0.25), ("roll", 0.2, 0.0, 0.1), ("roll", 0.2, 0.0, 0.5),
                                  ("pitch", 0.2, 0.0, 0.5)):  # fmt: skip
        out.append(Cond(f"C.sway.walk.{axis}{amp:g}@{hz:g}Hz", "C", "pose",
                        {"pose": {"kind": "sway", "axis": axis, "amp": amp, "center": center, "hz": hz}}, n=n))  # fmt: skip
    return out


def s2_conditions(n: int = 512) -> list[Cond]:
    """The Stage-2 programs (visibility gates + stop-go with pose)."""
    out: list[Cond] = []
    for axis, v in (("roll", 0.2), ("roll", -0.2), ("pitch", 0.2), ("pitch", -0.2), ("pitch", -0.15), ("roll", 0.15),
                    ("roll", -0.15), ("z", -0.020), ("z", 0.008)):  # fmt: skip
        tag = f"{axis}{v:+g}"
        out.append(
            Cond(
                f"S.lean.{tag}",
                "S",
                "s2lean",
                {"axis": axis, "value": v, "release": "ramp"},
                n=n,
                seconds_=12.5,
            )
        )
    out.append(Cond("S.lean.roll+0.2snap", "S", "s2lean", {"axis": "roll", "value": 0.2, "release": "snap"}, n=n,
                    seconds_=12.5))  # fmt: skip
    out.append(
        Cond(
            "S.head.pullback",
            "S",
            "s2head",
            {"mode": "pullback"},
            n=n,
            head_alpha=1.0,
            seconds_=8.0,
        )
    )
    out.append(
        Cond("S.head.micro", "S", "s2head", {"mode": "micro"}, n=n, head_alpha=1.0, seconds_=13.0)
    )
    for v in (0.40, 0.55):
        out.append(Cond(f"S.gaze.v{round(v * 100)}.home", "S", "s2gaze", {"vx": v, "head": "home"}, n=n, head_alpha=1.0,
                        seconds_=16.5))  # fmt: skip
    for yaw in (1.0, -1.0):
        out.append(Cond(f"S.gaze.v40.y{yaw:+g}", "S", "s2gaze", {"vx": 0.40, "head": "static", "yaw": yaw}, n=n,
                        head_alpha=1.0, seconds_=16.5))  # fmt: skip
    out.append(
        Cond(
            "S.gaze.v40.sacc",
            "S",
            "s2gaze",
            {"vx": 0.40, "head": "sacc"},
            n=n,
            head_alpha=1.0,
            seconds_=16.5,
        )
    )
    for v in (0.40, 0.55):
        out.append(
            Cond(
                f"S.stopgo.v{round(v * 100)}",
                "S",
                "s2stopgo",
                {"vx": v},
                n=n,
                head_alpha=1.0,
                seconds_=30.0,
            )
        )
    return out


QUICK_KEYS = (
    "S.lean.roll+0.2", "S.lean.pitch-0.2", "S.lean.pitch-0.15", "S.lean.roll+0.15", "S.lean.z-0.02",
    "S.lean.roll+0.2snap", "S.head.pullback", "S.head.micro", "S.gaze.v40.home", "S.gaze.v55.home",
    "S.gaze.v40.y+1", "S.stopgo.v40", "C.sway.walk.roll0.2@0.25Hz", "C.idle.v5.quiet", "C.push.v5.stand",
    "D.yaw_ou", "D.startle", "A.v40.home",
)  # fmt: skip
CORE_KEYS = (
    *QUICK_KEYS, "S.lean.roll-0.2", "S.lean.pitch+0.2", "S.lean.roll-0.15", "S.lean.z+0.008", "S.gaze.v40.y-1",
    "S.gaze.v40.sacc", "S.stopgo.v55", "C.trans.v5.v40", "C.trans.v5.v55", "C.push.v5.walk", "C.posture.v5",
    "A.v55.home", "D.alive", "C.sway.walk.roll0.2@0.5Hz", "C.sway.walk.pitch0.2@0.25Hz",
)  # fmt: skip


V5CTL_REG_KEYS = ("C.idle.v5.quiet", "C.idle.v5.lively", "A.v40.home", "A.v55.home", "D.yaw_ou", "C.push.v5.stand",
                  "C.push.v5.walk")  # fmt: skip


def conditions(mode: str, n: int) -> list[Cond]:
    reg = regression_conditions(n)
    s2 = s2_conditions(n)
    if mode == "full":
        return reg + s2
    if mode == "core":
        return [c for c in reg + s2 if c.key in CORE_KEYS]
    if mode == "quick":
        return [c for c in reg + s2 if c.key in QUICK_KEYS]
    if mode == "v5ctl":
        # v5 on the Stage-2 programs and on every regression condition a relative gate compares with,
        # so each servo delay has its own v5 reference (the stored A100 run has the recipe's delay only)
        return s2 + [c for c in reg if c.key in V5CTL_REG_KEYS]
    raise ValueError(f"mode {mode!r}")


# ── the Stage-2 programs (neuro_a100.programs.Program subclasses) ───────────────────────────


def _programs() -> dict[str, Any]:
    N = _neuro()
    P = N["P"]
    import torch

    HZ, DT = P.HZ, P.DT

    def rate_limit(cur: Any, goal: Any) -> Any:
        rate = (
            torch.tensor([C.POSE.rate_m_s, C.POSE.rate_rad_s, C.POSE.rate_rad_s], device=cur.device)
            * DT
        )
        return cur + (goal - cur).clamp(-rate, rate)

    class _Extra:
        """Per-tick channels a program records itself (after each step): each foot's vertical
        floor force, the head camera's world yaw and pitch."""

        def _extra_setup(self, T: int) -> None:
            self.fz = torch.zeros(T, self.n, 2, device=self.device)
            self.cam_yaw = torch.zeros(T, self.n, device=self.device)
            self.cam_pitch_w = torch.zeros(T, self.n, device=self.device)

        def after_step(self, k: int, env: Any) -> None:
            if k >= self.fz.shape[0]:
                return
            robot = env.scene["robot"]
            s = self.sl
            sensor = env.scene.sensors["feet_ground_contact"]
            f = sensor.data.force
            if f is not None:
                f = f.reshape(env.num_envs, -1, 3)[s, :2, 2].abs()
                self.fz[k] = f
            cache = env.__dict__.setdefault("_s2b_cam", None)
            if cache is None:
                ids, _ = robot.find_sites(["head_camera"], preserve_order=True)
                env._s2b_cam = cache = ids[0]
            q = robot.data.site_quat_w[s, cache]
            w, x, y, z = q.unbind(-1)
            ax = torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=1
            )
            self.cam_yaw[k] = torch.atan2(ax[:, 1], ax[:, 0])
            self.cam_pitch_w[k] = torch.rad2deg(torch.asin(ax[:, 2].clamp(-1, 1)))

    class Lean(_Extra, P.Program):
        """robot.pose ramped in (duck-body's slew), held, released by a ramp or at once."""

        ON, HOLD_S, END_AFTER = 3.0, 4.5, 2.5

        def setup(self) -> None:
            p = self.params
            self.off_t = self.ON + self.HOLD_S
            self.seconds = self.off_t + 1.0 + self.END_AFTER
            self._extra_setup(int(round((self.seconds + 1) * HZ)))
            idx = {"z": 0, "roll": 1, "pitch": 2}[p["axis"]]
            self.goal = torch.zeros(self.n, 3, device=self.device)
            self.goal[:, idx] = float(p["value"])
            self.cur = torch.zeros(self.n, 3, device=self.device)
            self.released_at = torch.full((self.n,), -1, dtype=torch.long, device=self.device)

        def targets(self, k: int, st: Any) -> None:
            t = self.t(k)
            s = self.sl
            if self.ON <= t < self.off_t:
                self.cur = rate_limit(self.cur, self.goal)
                st.pose_on[s] = True
                st.body[s] = self.cur
            elif t >= self.off_t:
                if self.params.get("release") == "ramp":
                    self.cur = rate_limit(self.cur, torch.zeros_like(self.cur))
                    on = self.cur.abs().amax(dim=1) > 1e-6
                else:
                    on = torch.zeros(self.n, dtype=torch.bool, device=self.device)
                    self.cur = torch.zeros_like(self.cur)
                newly = ~on & (self.released_at < 0)
                self.released_at = torch.where(
                    newly, torch.full_like(self.released_at, k), self.released_at
                )
                st.pose_on[s] = on
                st.body[s] = torch.where(on.unsqueeze(1), self.cur, torch.zeros_like(self.cur))

        def metrics(self, tr: dict[str, Any], T: int) -> dict[str, list]:
            p = self.params
            on, off = P._i(self.ON, T), P._i(self.off_t, T)
            b0, h0 = P._i(self.ON - 1.0, T), P._i(self.off_t - 1.5, T)
            e = P._i(self.seconds, T)
            m = {"pre_bad": (tr["fallen"][:on] > 0.5).any(dim=0)}
            m.update(P.safety(tr, on, e))
            ser = {
                "roll": torch.rad2deg(tr["roll"]),
                "pitch": torch.rad2deg(tr["pitch"]),
                "z": 1000 * tr["z"],
            }[p["axis"]]
            base = ser[b0:on].mean(dim=0)
            ach = ser[h0:off].mean(dim=0) - base
            unit = 1000.0 if p["axis"] == "z" else math.degrees(1.0)
            cmd = float(p["value"]) * unit
            m["achieved"] = ach
            m["gain"] = ach / cmd
            sgn = 1.0 if cmd >= 0 else -1.0
            idx = P.first_idx(sgn * (ser[on:off] - base) >= 0.9 * (sgn * ach).clamp_min(1e-3))
            m["t90_s"] = torch.where(
                idx >= 0, (idx + 1).float() * DT, torch.full_like(idx, -1).float()
            )
            m["steps_pose"] = tr["td"][on:off].sum(dim=0)
            rel = self.released_at.clamp(min=0)
            gy = tr["gyro"]
            win = torch.stack([gy[(rel + j).clamp(max=T - 1), torch.arange(self.n, device=self.device)]
                               for j in range(1, HZ + 1)], dim=0)  # fmt: skip
            m["release_rate"] = win.max(dim=0).values
            fz = self.fz[h0:off]
            tot = fz.sum(dim=2).clamp_min(1e-3)
            m["load_share"] = (fz.max(dim=2).values / tot).mean(dim=0)
            m["load_right"] = (fz[:, :, 1] / tot).mean(dim=0)
            m.update(P.effort(tr, h0, off))
            return P.tolist(m)

    class Head(_Extra, P.Program):
        """Standing: the neck pull-back held 2.5 s, or ±0.052 rad (±3°) of head yaw at 1 Hz for 8 s."""

        ON = 3.0

        def setup(self) -> None:
            self.mode = self.params["mode"]
            self.off_t = self.ON + (2.5 if self.mode == "pullback" else 8.0)
            self.seconds = self.off_t + (2.5 if self.mode == "pullback" else 2.0)
            self._extra_setup(int(round((self.seconds + 1) * HZ)))

        def targets(self, k: int, st: Any) -> None:
            t = self.t(k)
            s = self.sl
            if self.ON <= t < self.off_t:
                if self.mode == "pullback":
                    st.head[s, 0] = 0.3
                    st.head[s, 1] = -0.2
                else:
                    st.head[s, 2] = 0.052 * math.sin(2 * math.pi * 1.0 * (t - self.ON))

        def metrics(self, tr: dict[str, Any], T: int) -> dict[str, list]:
            on, off, e = P._i(self.ON, T), P._i(self.off_t, T), P._i(self.seconds, T)
            m = {"pre_bad": (tr["fallen"][:on] > 0.5).any(dim=0)}
            m.update(P.safety(tr, on, e))
            if self.mode == "pullback":
                b0, h0 = P._i(self.ON - 0.5, T), P._i(self.off_t - 1.0, T)
                m["neck_gain"] = (
                    tr["hq_neck"][h0:off].mean(dim=0) - tr["hq_neck"][b0:on].mean(dim=0)
                ) / 0.3
                m["head_gain"] = (
                    tr["hq_pitch"][h0:off].mean(dim=0) - tr["hq_pitch"][b0:on].mean(dim=0)
                ) / -0.2
                m["cam_up"] = self.cam_pitch_w[h0:off].mean(dim=0) - self.cam_pitch_w[b0:on].mean(
                    dim=0
                )
            else:
                i0 = P._i(self.ON + 2.0, T)
                t = (
                    torch.arange(i0, off, device=self.device, dtype=torch.float32) * DT
                    + DT
                    - self.ON
                )
                amp, lag = P.lsq_sine(tr["hq_yaw"][i0:off], t, 1.0)
                m["gain"] = amp / 0.052
                m["lag_ms"] = 1000 * lag / (2 * math.pi)
            m["steps"] = tr["td"][on:e].sum(dim=0)
            m.update(P.effort(tr, on, off))
            return P.tolist(m)

    class Gaze(_Extra, P.Program):
        """Walking at vx with the head at HOME, a held yaw offset or saccades (head_alpha 1.0)."""

        WALK_ON, HEAD_ON, STOP, END = 2.0, 3.0, 14.0, 16.5
        WIN = (5.0, 14.0)

        def setup(self) -> None:
            self.seconds = self.END
            self._extra_setup(int(round((self.seconds + 1) * HZ)))
            if self.params.get("head") == "sacc":
                self.sac = P.Saccades(self.n, 0.3, 0.8, 0.7, (0.8, 3.0), self.gen, self.device)

        def targets(self, k: int, st: Any) -> None:
            t = self.t(k)
            s = self.sl
            if self.WALK_ON <= t < self.STOP:
                st.twist[s, 0] = self.params["vx"]
            if t >= self.HEAD_ON and t < self.STOP:
                h = self.params.get("head")
                if h == "static":
                    st.head[s, 2] = float(self.params["yaw"])
                elif h == "sacc":
                    st.head[s, 2] = self.sac.step()

        def metrics(self, tr: dict[str, Any], T: int) -> dict[str, list]:
            i0, i1 = P._i(self.WIN[0], T), P._i(self.WIN[1], T)
            a0, e = P._i(self.WALK_ON, T), P._i(self.END, T)
            m = {"pre_bad": (tr["fallen"][:a0] > 0.5).any(dim=0)}
            m.update(P.safety(tr, a0, e))
            m["speed"] = P.path_speed(tr, i0, i1)
            m["cam_perp"] = torch.sqrt((tr["cam_perp"][i0:i1] ** 2).mean(dim=0))
            cy = P.unwrap_yaw(self.cam_yaw[i0:i1])
            ty = P.unwrap_yaw(tr["yaw"][i0:i1])
            m["head_yaw_sd"] = torch.rad2deg(_highpass_sd(cy))
            m["trunk_yaw_sd"] = torch.rad2deg(_highpass_sd(ty))
            m["cam_pitch_sd"] = _highpass_sd(self.cam_pitch_w[i0:i1])
            m["cam_pitch"] = self.cam_pitch_w[i0:i1].mean(dim=0)
            hq = torch.stack([tr["hq_neck"], tr["hq_pitch"], tr["hq_yaw"], tr["hq_roll"]], dim=2)[
                i0:i1
            ]
            m["neck_travel_min"] = hq.diff(dim=0).abs().sum(dim=(0, 2)) / ((i1 - i0) * DT / 60.0)
            if self.params.get("head") == "static":
                m["yaw_delivered"] = tr["hq_yaw"][i0:i1].mean(dim=0) / float(self.params["yaw"])
            m.update(P.effort(tr, i0, i1))
            return P.tolist(m)

    def _highpass_sd(x: Any, k: int = 25) -> Any:
        """SD over time of x minus its centred k-tick moving mean (the stride band, > ~1 Hz),
        over the interior samples (no edge padding: a slow turn leaves no residue)."""
        pad = k // 2
        cs = torch.cumsum(torch.cat([torch.zeros_like(x[:1]), x], dim=0), dim=0)
        mean = (cs[k:] - cs[:-k]) / k  # the mean of x[i : i + k], centred on i + pad
        return (x[pad : pad + mean.shape[0]] - mean).std(dim=0)

    class StopGo(_Extra, P.Program):
        """3 × (walk 3 s → stop → wait 0 / 0.7 / 1.5 s → pose (roll 0.15 sine 0.3 Hz, pitch −0.1) 3 s →
        ramp release → 0.5 s) then 3 s standing: the stop-go mode where alpha_stand's handover failed."""

        WAITS = (0.0, 0.7, 1.5)

        def setup(self) -> None:
            self._extra_setup(int(round(31 * HZ)))
            sched, t = [], 2.0
            for w in self.WAITS:
                walk = (t, t + 3.0)
                pose = (walk[1] + w, walk[1] + w + 3.0)
                sched.append((walk, pose))
                t = pose[1] + 0.5 + 0.5  # ramp-out ≤ 0.3 s at 0.5 rad/s, then 0.5 s
            self.sched = sched
            self.seconds = t + 3.0
            self.cur = torch.zeros(self.n, 3, device=self.device)

        def targets(self, k: int, st: Any) -> None:
            t = self.t(k)
            s = self.sl
            for (w0, w1), (p0, p1) in self.sched:
                if w0 <= t < w1:
                    st.twist[s, 0] = self.params["vx"]
                if p0 <= t < p1:
                    goal = torch.zeros(self.n, 3, device=self.device)
                    goal[:, 1] = 0.15 * math.sin(2 * math.pi * 0.3 * (t - p0))
                    goal[:, 2] = -0.10
                    self.cur = rate_limit(self.cur, goal)
                    st.pose_on[s] = True
                    st.body[s] = self.cur
                elif p1 <= t < p1 + 0.5:
                    self.cur = rate_limit(self.cur, torch.zeros_like(self.cur))
                    on = self.cur.abs().amax(dim=1) > 1e-6
                    st.pose_on[s] = on
                    st.body[s] = torch.where(on.unsqueeze(1), self.cur, torch.zeros_like(self.cur))

        def metrics(self, tr: dict[str, Any], T: int) -> dict[str, list]:
            a0, e = P._i(2.0, T), P._i(self.seconds, T)
            m = {"pre_bad": (tr["fallen"][:a0] > 0.5).any(dim=0)}
            m.update(P.safety(tr, a0, e))
            steps = torch.zeros(self.n, device=self.device)
            for _w, (p0, p1) in self.sched:
                steps += tr["td"][P._i(p0, T) : P._i(p1, T)].sum(dim=0)
            m["steps_pose"] = steps
            v = P.tick_speed(tr)
            stops = []
            for (_w0, w1), _p in self.sched:
                i = P._i(w1, T)
                idx = P.first_idx(v[i : min(e, i + 100)] < P.STILL_MPS)
                stops.append(
                    torch.where(idx >= 0, (idx + 1).float() * DT, torch.full_like(idx, -1).float())
                )
            m["stop_s"] = torch.stack(stops, dim=1)
            m.update(P.effort(tr, a0, e))
            return P.tolist(m)

    return {"s2lean": Lean, "s2head": Head, "s2gaze": Gaze, "s2stopgo": StopGo}


def summarize_s2(kind: str, params: dict, rows_all: list[dict]) -> dict[str, Any]:
    """Compact summaries of the Stage-2 programs (neuro_a100.report's formats)."""
    rep = _neuro()["rep"]
    S = __import__("neuro_a100.stats", fromlist=["x"])
    rows, excluded = rep.valid(rows_all)
    s = rep.common(rows, excluded)
    v = lambda k: rep._vals(rows, k)  # noqa: E731
    if kind == "s2lean":
        s["ach"] = S.spread5(v("achieved"), 3)
        s["gain"] = S.spread5(v("gain"), 3)
        s["t90"] = S.spread(rep._pos(v("t90_s")), 3)
        s["st_pose"] = rep._rate_true(rows, "steps_pose")
        s["rrate"] = S.spread(v("release_rate"), 1)
        s["share"] = S.spread5(v("load_share"), 3)
        s["right"] = S.spread5(v("load_right"), 3)
    elif kind == "s2head":
        for k in ("neck_gain", "head_gain", "cam_up", "gain", "lag_ms"):
            if any(k in r for r in rows):
                s[k] = S.spread5(v(k), 3)
        s["stepped"] = rep._rate_true(rows, "steps")
    elif kind == "s2gaze":
        s["spd"] = S.mean_ci(v("speed"))
        s["camx"] = S.spread(v("cam_perp"), 1)
        s["hysd"] = S.spread(v("head_yaw_sd"), 3)
        s["tysd"] = S.spread(v("trunk_yaw_sd"), 3)
        s["cpsd"] = S.spread(v("cam_pitch_sd"), 3)
        s["camp"] = S.mean_ci(v("cam_pitch"), 2)
        s["travel"] = S.spread(v("neck_travel_min"), 1)
        if any("yaw_delivered" in r for r in rows):
            s["deliv"] = S.spread5(v("yaw_delivered"), 3)
    elif kind == "s2stopgo":
        s["st_pose"] = rep._rate_true(rows, "steps_pose")
        s["stop"] = S.spread(rep._pos(v("stop_s")), 2)
    s["eff"] = rep.effort(rows)
    return s


# ── the environment (the neuro validation's, with a chosen servo delay) ─────────────────────


def delay_steps(label: str) -> tuple[int, int] | None:
    """``j3-6`` → None (the recipe's), ``d20`` → (4, 4)."""
    if label.startswith("j"):
        lo, _, hi = label[1:].partition("-")
        return (int(lo), int(hi)) if (lo, hi) != ("3", "6") else None
    ms = float(label[1:])
    s = int(round(ms / 5.0))
    if abs(s * 5.0 - ms) > 1e-9:
        raise ValueError(f"delay {label}: not a multiple of 5 ms")
    return (s, s)


def build_env(envs: int, device: str, episode_s: float, delay: str = "j3-6", seed: int = 0) -> Any:
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    N = _neuro()
    sim = N["sim"]
    cfg = load_env_cfg(sim.TASK, play=True)
    cfg.scene.num_envs = envs
    cfg.seed = seed
    cfg.episode_length_s = episode_s
    cfg.curriculum = {}
    for name in ("push_robot", "topple_push", "random_prone_init"):
        cfg.events.pop(name, None)
    for name, command in cfg.commands.items():
        command.resampling_time_range = (1e9, 1e9)
        if name in ("head_pose", "body_pose"):
            command.ranges = ((0.0, 0.0),) * len(command.ranges)
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = twist.rel_forward_envs = twist.rel_turn_in_place_envs = 0.0
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0.0, 0.0)
    if "set_ground_state" not in cfg.events:
        cfg.events["set_ground_state"] = deepcopy(
            load_env_cfg(sim.SIT_TASK, play=True).events["set_ground_state"]
        )
    cfg.events["set_ground_state"].params.update(sitting_prob=0.0, standing_prob=1.0)
    if "randomize_com" in cfg.events:
        cfg.events["randomize_com"].params["ranges"] = (-sim.TRUNK_COM_M, sim.TRUNK_COM_M)
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["ranges"] = (-sim.HEAD_COM_M, sim.HEAD_COM_M)
    for name in list(cfg.terminations):
        if name != "nan_state":
            cfg.terminations.pop(name)
    steps = delay_steps(delay)
    if steps is not None:
        import dataclasses

        robot = deepcopy(cfg.scene.entities["robot"])
        acts = tuple(dataclasses.replace(a, delay_min_lag=steps[0], delay_max_lag=steps[1], delay_hold_prob=0.0,
                                         delay_update_period=0) for a in robot.articulation.actuators)  # fmt: skip
        robot.articulation = dataclasses.replace(robot.articulation, actuators=acts)
        cfg.scene.entities = {**cfg.scene.entities, "robot": robot}
    return ManagerBasedRlEnv(cfg, device=device)


# ── the policies ─────────────────────────────────────────────────────────────────────────────


def walk_policy(source: str, device: str) -> tuple[Any, dict[str, Any]]:
    """A batched walk policy from an ONNX (torch reconstruction) or a training checkpoint's actor."""
    from duckmind.skilltrain import rl_onnx

    if source.endswith(".onnx"):
        fn = rl_onnx.torch_policy(source, device)
        return fn, {"onnx": Path(source).name, "sha256": rl_onnx.sha256(source)}
    from duckmind.skilltrain.rl_eval import checkpoint_policy

    fn, info = checkpoint_policy(source, device)
    return fn, info


# ── the run ─────────────────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    N = _neuro()
    NR, NRd, NPol, NS, rep = N["run"], N["R"], N["P"], N["sim"], N["rep"]
    from neuro_a100 import policy as P5

    NPol.KINDS.update(_programs())
    torch.set_grad_enabled(False)
    t0 = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s2b {time.monotonic() - t0:7.1f}s] {msg}", flush=True)

    device = args.device
    policies = Path(args.policies)
    P5.ensure_policies(policies, None, log=log)
    sit = P5.load_policy(policies / "alpha_sitstand.onnx", device)
    arms: dict[str, tuple[Any, dict]] = {}
    if args.mode == "v5ctl":
        v5 = P5.load_policy(policies / "velstand.onnx", device)
        arms["v5"] = (v5, {"onnx": "velstand.onnx (v5)"})
    else:
        fn, info = walk_policy(args.walk, device)
        arms["cand"] = (fn, info)
    delays = [d for d in args.delays.split(",") if d]
    n = args.n if args.n else (48 if args.mode == "quick" else 512)
    plans: list[tuple[str, list[Cond]]] = []
    for i, d in enumerate(delays):
        if args.mode == "full":
            conds = conditions("full" if i == 0 else "core", n)
        else:
            conds = conditions(args.mode, n)
        if args.only:
            conds = [c for c in conds if any(c.key.startswith(o) for o in args.only.split(","))]
        plans.append((d, conds))
    envs = args.envs or (NR.auto_envs(device) if device.startswith("cuda") else 64)
    result: dict[str, Any] = {
        "format": FORMAT,
        "complete": False,
        "errors": [],
        "meta": {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": args.mode,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
            "arms": {k: v[1] for k, v in arms.items()},
            "delays": delays,
            "n": n,
            "envs": envs,
            "stage": args.stage,
            "harness": "neuro_a100 (A100 neuro validation 2026-09-28) + s2 programs",
        },  # fmt: skip
        "arms": {},
    }
    out = Path(args.out)
    rows_path = out.with_suffix(".rows.jsonl.gz")
    with gzip.open(rows_path, "wt") as fh:
        fh.write(json.dumps({"header": result["meta"]}) + "\n")

    def save() -> None:
        result["meta"]["seconds"] = round(time.monotonic() - t0, 1)
        out.write_text(json.dumps(rep.clean(result), ensure_ascii=True, separators=(",", ":")))

    for delay, conds in plans:
        episode = max(c.seconds for c in conds) + 20
        env = build_env(envs, device, episode, delay)  # neuro_a100.sim.build_env + the chosen delay
        reader, peaks = NS.Reader(env), NS.SubstepPeaks(env)
        by_key = {c.key: c for c in conds}
        for arm, (fn, _info) in arms.items():
            nets = {NRd.NET_WALK: fn, NRd.NET_SITSTAND: sit}
            batches = NR.pack(conds, envs)
            rows_by_key: dict[str, list[dict]] = {}
            log(
                f"{arm} @ {delay}: {len(conds)} conditions, {sum(c.n for c in conds)} ducks, {len(batches)} batches of ≤ {envs}"
            )
            for bi, batch in enumerate(batches, 1):
                try:
                    rows, info = NR.run_batch(
                        env, nets, batch, by_key, device, 1000 + bi, reader, peaks
                    )
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    log(f"batch {bi} failed:\n{tb}")
                    result["errors"].append(
                        {"arm": arm, "delay": delay, "batch": bi, "traceback": tb[-3000:]}
                    )
                    continue
                with gzip.open(rows_path, "at") as fh:
                    for key, rs in rows.items():
                        rows_by_key.setdefault(key, []).extend(rs)
                        for r in rs:
                            fh.write(
                                json.dumps(
                                    rep.clean({**r, "arm": arm, "delay": delay}),
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                log(
                    f"  batch {bi}/{len(batches)}: {batch.used} ducks × {info['ticks']} ticks in {info['sim_s']:.0f} s"
                )
            blocks: dict[str, Any] = {}
            for c in conds:
                rs = rows_by_key.get(c.key)
                if not rs:
                    continue
                try:
                    s = (
                        summarize_s2(c.kind, c.params, rs)
                        if c.kind.startswith("s2")
                        else rep.summarize(c.kind, c.params, rs)
                    )
                except Exception as exc:  # noqa: BLE001
                    s = {"error": f"{type(exc).__name__}: {exc}"}
                s["walkoff"] = _walkoffs(rs) if c.kind == "stand_idle" else None
                blocks[c.key] = s
            result["arms"].setdefault(arm, {})[delay] = blocks
            save()
        NS.release(env)
    result["complete"] = not result["errors"]
    if args.mode != "v5ctl":
        result["gates"] = evaluate(
            result, args.stage, v5_results=args.v5_results, v5_ctl=args.v5_ctl
        )
    save()
    log(f"wrote {out} (complete={result['complete']})")
    return result


def _walkoffs(rows: list[dict]) -> list[Any]:
    """Ducks that moved > 40 cm during a zero-command stand, [k, n]."""
    vals = [r.get("drift_cm") for r in rows if not r.get("reset") and not r.get("pre_bad")]
    vals = [v for v in vals if v is not None]
    return [sum(1 for v in vals if v > 100 * 0.40), len(vals)]


# ── the gates ────────────────────────────────────────────────────────────────────────────────


def _p50(spread: Any, i: int = 0) -> float | None:
    return None if not spread else spread[i]


def _k_n(rate_: Any) -> tuple[int, int] | None:
    return None if not rate_ else (int(rate_[0]), int(rate_[1]))


def _pool(blocks: dict[str, Any], keys: list[str], field_: str) -> tuple[int, int] | None:
    k = n = 0
    for key in keys:
        r = (blocks.get(key) or {}).get(field_)
        if r:
            k += int(r[0])
            n += int(r[1])
    return (k, n) if n else None


def evaluate(
    result: dict[str, Any], stage: str, *, v5_results: str | None = None, v5_ctl: str | None = None
) -> dict[str, Any]:
    """Every gate of :mod:`s2_contract` for the candidate at each delay, with v5's stored numbers
    (the 2026-09-28 A100 run) and, where the stored run has none, the v5 control battery."""
    stored = {}
    path = Path(v5_results) if v5_results else V5_RESULTS_DEFAULT
    if path.is_file():
        stored = json.loads(path.read_text()).get("blocks", {})
    v5b = {k: v for blk in stored.values() for k, v in blk.items()}
    ctl_all: dict[str, Any] = {}
    if v5_ctl and Path(v5_ctl).is_file():
        ctl_all = json.loads(Path(v5_ctl).read_text()).get("arms", {}).get("v5") or {}
    stored_v5b = v5b
    out: dict[str, Any] = {}
    for delay, b in (result.get("arms", {}).get("cand") or {}).items():
        g: dict[str, Any] = {}
        # v5 at the same servo delay when the control battery has it (else the recipe's jitter)
        ctl = ctl_all.get(delay) or ctl_all.get("j3-6", {})
        v5b = {**stored_v5b, **ctl}
        g["_v5_reference"] = {"delay": delay if delay in ctl_all else ("j3-6" if ctl_all else None),
                              "stored": "A100 2026-09-28 (the recipe's 15–30 ms jitter)"}  # fmt: skip

        def put(key: str, value: Any, detail: Any = None, v5: Any = None) -> None:
            gate = C.GATE_BY_KEY[key]
            g[key] = {"value": value, "verdict": C.verdict(gate, value, v5), "threshold": gate.threshold,
                      "detail": detail, "zh": gate.zh}  # fmt: skip

        lean = lambda k: _p50((b.get(k) or {}).get("ach"), 1)  # noqa: E731
        vals = [lean("S.lean.roll+0.2"), lean("S.lean.roll-0.2")]
        vals = [abs(v) for v in vals if v is not None]
        put("lean_roll_deg", min(vals) if vals else None, {"roll±0.2 p50": vals})
        vals = [lean("S.lean.pitch+0.2"), lean("S.lean.pitch-0.2")]
        vals = [abs(v) for v in vals if v is not None]
        put("lean_pitch_deg", min(vals) if vals else None, {"pitch±0.2 p50": vals})
        sways = [
            k for k in b if k.startswith("C.sway.walk.") and ("roll0.2" in k or "pitch0.2" in k)
        ]
        amps = {
            k: round(_p50(b[k].get("gain"), 1) * math.degrees(0.2), 2)
            for k in sways
            if b[k].get("gain")
        }
        lags = {k: _p50(b[k].get("lag"), 1) for k in sways if b[k].get("lag")}
        put("sway_amp_deg", min(amps.values()) if amps else None, amps)
        put("sway_lag_ms", max(lags.values()) if lags else None, lags)
        pose_keys = [
            k for k in b if k.startswith(("S.lean.", "C.pose.walk.", "C.sway.walk.", "C.pose.v5."))
        ]
        put("pose_stepped", _pool(b, pose_keys, "st_pose"), {"conditions": len(pose_keys)})
        put(
            "release_rate_dps",
            _p50((b.get("S.lean.roll+0.2snap") or {}).get("rrate")),
            {"ramped roll+0.2": _p50((b.get("S.lean.roll+0.2") or {}).get("rrate"))},
        )
        shares = [
            _p50((b.get(k) or {}).get("share"), 1) for k in ("S.lean.roll+0.15", "S.lean.roll-0.15")
        ]
        shares = [s for s in shares if s is not None]
        put("weight_shift_ratio", min(shares) if shares else None, shares)
        chest = lean("S.lean.pitch-0.15")
        put("chest_out_deg", -chest if chest is not None else None)
        crouch = lean("S.lean.z-0.02")
        put("crouch_mm", -crouch if crouch is not None else None)
        put("neck_pullback_gain", _p50((b.get("S.head.pullback") or {}).get("neck_gain"), 1))
        put("yaw_micro_gain", _p50((b.get("S.head.micro") or {}).get("gain"), 1))
        lrms = ((b.get("C.idle.v5.quiet") or {}).get("eff") or {}).get("lrms")
        base_lrms = ((v5b.get("C.idle.v5.quiet") or {}).get("eff") or {}).get("lrms") or C.V5[
            "stand_leg_rms_nm"
        ]
        put(
            "stand_leg_rms_ratio",
            round(lrms / base_lrms, 4) if lrms else None,
            {"cand": lrms, "v5": base_lrms},
        )
        put("pose_falls", _pool(b, pose_keys, "fall"))
        # gaze
        ratios, sd_ratios = {}, {}
        for v, sv in (("40", "0.40"), ("55", "0.55")):
            cx = _p50((b.get(f"S.gaze.v{v}.home") or {}).get("camx"))
            base = (
                _p50((ctl.get(f"S.gaze.v{v}.home") or {}).get("camx")) or C.V5["cam_perp_dps"][sv]
            )
            if cx is not None:
                ratios[sv] = round(cx / base, 3)
            hs = _p50((b.get(f"S.gaze.v{v}.home") or {}).get("hysd"))
            hb = _p50((ctl.get(f"S.gaze.v{v}.home") or {}).get("hysd"))
            if hs is not None and hb:
                sd_ratios[sv] = round(hs / hb, 3)
        if stage == "gaze":
            put("cam_perp_ratio", max(ratios.values()) if ratios else None, ratios)
            put(
                "head_yaw_stride_sd_ratio",
                max(sd_ratios.values()) if sd_ratios else None,
                sd_ratios,
            )
            dl = [
                _p50((b.get(k) or {}).get("deliv"), 1) for k in ("S.gaze.v40.y+1", "S.gaze.v40.y-1")
            ]
            dl = [d for d in dl if d is not None]
            put("head_yaw_delivered", min(dl) if dl else None, dl)
            put("neck_travel_rad_min", _p50((b.get("S.gaze.v40.home") or {}).get("travel")))
        else:
            put("cam_perp_regress_ratio", max(ratios.values()) if ratios else None, ratios)
        # regression
        walk_keys = [k for k in b if k.startswith(("A.", "B.walk.", "D.", "S.gaze."))]
        put("walk_falls", _pool(b, walk_keys, "fall"), {"conditions": len(walk_keys)})
        sg_keys = [k for k in b if k.startswith(("C.trans.", "S.stopgo."))]
        put("stopgo_near_falls", _pool(b, sg_keys, "t45"), {"conditions": sg_keys})
        pushes = [k for k in b if k.startswith("C.push.")]
        try1 = _pool({k: (b[k].get("right") or {}) for k in pushes}, pushes, "try1")
        put("push_righting_8s", try1)
        rt = [_p50(((b.get(k) or {}).get("right") or {}).get("t")) for k in pushes]
        rtb = [_p50(((v5b.get(k) or {}).get("right") or {}).get("t")) for k in pushes]
        extra = [a - c for a, c in zip(rt, rtb, strict=False) if a is not None and c is not None]
        put(
            "righting_p50_extra_s", round(max(extra), 3) if extra else None, {"cand": rt, "v5": rtb}
        )
        put("sitstand_falls", _k_n((b.get("C.posture.v5") or {}).get("fall")))
        sp = {}
        for v in ("40", "55"):
            a = (b.get(f"A.v{v}.home") or {}).get("spd")
            c0 = (v5b.get(f"A.v{v}.home") or {}).get("spd")
            if a and c0:
                sp[v] = round(a[0] / c0[0], 4)
        put("walk_speed_ratio", max(sp.values(), key=lambda x: abs(x - 1)) if sp else None, sp)
        yg = (b.get("D.yaw_ou") or {}).get("ygain")
        ygb = (v5b.get("D.yaw_ou") or {}).get("ygain")
        put(
            "turn_gain_ratio",
            round(yg[1] / ygb[1], 4) if yg and ygb else None,
            {"cand": yg, "v5": ygb},
        )
        put("stop_s_p50", _p50((b.get("D.startle") or {}).get("stop")))
        a_keys = [k for k in b if k.startswith("A.") and not k.endswith(".home")]
        put("head_motion_falls", _pool(b, a_keys, "fall"), v5=C.V5["head_motion_falls"])
        put(
            "stand_stepped",
            _k_n((b.get("C.idle.v5.quiet") or {}).get("stepped")),
            v5=C.V5["stand_stepped_quiet"],
        )
        wo = (b.get("C.idle.v5.quiet") or {}).get("walkoff")
        wob = (ctl.get("C.idle.v5.quiet") or {}).get("walkoff")
        put("walkoffs", tuple(wo) if wo else None, v5=tuple(wob) if wob else None)
        out[delay] = g
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage-2 battery (candidate walk network vs v5, the plan's gates)"
    )
    p.add_argument(
        "--walk", default=None, help="the candidate: an ONNX or a training checkpoint (.pt)"
    )
    p.add_argument(
        "--policies",
        required=True,
        help="the v5 folder (velstand, alpha_sitstand, alpha_stand, manifest)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="quick", choices=("full", "core", "quick", "v5ctl"))
    p.add_argument("--delays", default="j3-6", help="comma list: j3-6 (the recipe's), d0, d20, d40")
    p.add_argument("--stage", default="pose", choices=C.STAGES)
    p.add_argument("--n", type=int, default=0, help="ducks per condition (0 = 512 full / 48 quick)")
    p.add_argument("--envs", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", default="")
    p.add_argument("--v5-results", default=None, help="the stored v5 results.json (the A100 run)")
    p.add_argument(
        "--v5-ctl", default=None, help="a v5ctl battery JSON (the Stage-2 programs on v5)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode != "v5ctl" and not args.walk:
        raise SystemExit("--walk is required (or --mode v5ctl)")
    if args.device.startswith("cpu"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    run(args)
    return 0


__all__ = ["Cond", "conditions", "evaluate", "run", "s2_conditions", "summarize_s2"]

if __name__ == "__main__":
    sys.exit(main())
