"""The headless go-there runner: the real brain on the real physics, one click at a time, no server and no clock.

Per click a fresh neural duck (:class:`duckmind.world.neuro_play_rig.PlayNeuro`: the whole brain, its senses from the
world's view geometry on the production camera, the motor, the body link, the official duck in the home with the shipped
walk) is stood at the click's start pose, given the known-home lasting map (:mod:`.knownhome`), sent to the place
(``brain.go_to``, a bearing and a range, as the page's click does) and watched for :data:`CLICK_S` seconds. The body is the
live simulated duck's: the walk policies and the collision model named on the command line (v8 on the backlash body), the
simulator's 20 ms servo command delay, the body link with that walk's measured map and proprio, turns on the spot handed
to the link, the ToF through the depth tracker at 15 Hz. Truth (the body's true pose, contacts) is read for scoring only.

Arms (what steers the legs during the errand; everything else is identical):

* ``main`` — the brain as it is.
* ``truth`` — the errand's own steering (``_errand_drive``: face the place, walk at it, the wall memory's bend and cap) is
  replaced by the truth planner's (:mod:`.truthplan`); every reflex (its stops, caps and bends), look-before-going, the
  cornered stop, the escapes, the selector's takeovers, the body link, the walk and the physics stay.
* ``truth-notake`` — ``truth``, and an errand that loses the walk to another act takes it back at once (the errand's
  ``REASSERT_S`` 3 s → 0): what the selector's takeovers cost. For information.
* ``truth-reflex`` — the planner has the legs for the whole errand; only the reflex layer still acts on them (a reflex's
  stop holds them, its cap and bend apply). Nothing else of the brain reaches the legs.
* ``truth-free`` — the planner has the legs for the whole errand and nothing of the brain reaches them (as a person's
  remote control: the body link's own gates stay). For information: what the body, the walk and the physics allow.

A row (schema :data:`SCHEMA`) is one click of one arm; see :data:`ROW_FIELDS`. Rows are deterministic: the same tree,
click and arm give the same row in any process.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields as dc_fields
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCHEMA = "bench-gothere-v1"
ARMS = ("main", "truth", "truth-notake", "truth-reflex", "truth-free")
CLICK_S = 40.0  # the errand's time: arrived within this, or not
SETTLE_S = 2.0
TOF_PERIOD_S = 1.0 / 15.0
ARRIVE_M = 0.18  # (truth scoring) as the errand's own arrival radius …
ARRIVE_HOLD_S = 0.3  # … held this long
TRUTH_OK_M = 0.25  # an arrival counts as true when the body ends within this of the place
COLLISION = "allcollisions_backlash"
COMMAND_DELAY_MS = "20"
LASTMAP_MODE = "soft"
WHEN = datetime(2026, 10, 24, 10, 0)
LOAD_MAX = 7.0
BATCHES_MAX = 2
# every row has these (the live mode fills what it can see and leaves the rest None)
ROW_FIELDS = (
    "schema", "set", "k", "arm", "mode", "tree", "seed", "pose", "point", "cls", "line", "flagged", "detour", "range_m", "bearing",
    "state", "reason", "arrived", "arrived_s", "truth_ok", "truth_in_s", "secs", "truth_end", "near", "falls", "bumps", "brushes",
    "startles", "gave_up", "takeovers", "escapes", "clutter_started", "ph", "rt_min",
)  # fmt: skip


def compose(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def inverse(a: Sequence[float]) -> tuple[float, float, float]:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2])


def prepare_env() -> None:
    """What the live simulated duck's processes have, set before anything of duckmind is imported in a worker."""
    os.environ.setdefault("DUCKMIND_LASTMAP", LASTMAP_MODE)
    os.environ.setdefault("DUCKMIND_BODY_COMMAND_DELAY_MS", COMMAND_DELAY_MS)


def _live_body(collision: str, policies: str) -> None:
    """The headless duck on the live duck's walk and collision model (process-wide, before any rig is built)."""
    from duckmind.world import headless_duck as HD
    from duckmind.world import neuro_play_rig as R

    if getattr(HD, "_bench_body", None) == (collision, policies):
        return
    if not hasattr(HD, "_bench_compose"):
        HD._bench_compose = HD.compose  # type: ignore[attr-defined]
        R.PlayDuck._bench_init = R.PlayDuck.__init__  # type: ignore[attr-defined]
    orig, init = HD._bench_compose, R.PlayDuck._bench_init  # type: ignore[attr-defined]

    def compose_as(home: Any, *a: Any, **k: Any) -> Any:
        k["collision"] = collision
        return orig(home, *a, **k)

    def init_as(self: Any, *a: Any, **k: Any) -> None:
        k["policies"] = policies
        init(self, *a, **k)

    HD.compose = compose_as  # type: ignore[assignment]
    R.PlayDuck.__init__ = init_as  # type: ignore[method-assign]
    HD._bench_body = (collision, policies)  # type: ignore[attr-defined]


def _link_as_live(rig: Any, collision: str, policies: str) -> dict[str, Any]:
    """The body link as the live driver sets it for this body (simweb.neuro_live._new_link)."""
    from duckmind.simweb import neuro_live as NL

    body = SimpleNamespace(paths=SimpleNamespace(policies=policies), collision=collision)
    link = rig.link.link
    link.walk = NL._walk_map(body, link.delay_ms)
    turned = NL._walk_proprio(body, link.proprio.cfg, link.delay_ms)
    seated = NL._seat_proprio(body, turned, link.delay_ms)
    sense, seat = seated if seated is not None else (turned, None)
    link.use_proprio(sense, seat=seat)
    return {
        "delay_ms": link.delay_ms,
        "walk_map": link.walk is not None,
        "pivots": bool(link._walk_pivots()),
        "seat": seat,
    }


_REFLEX_STEP: Any = None
_REASSERT_S: float | None = None


class Watch:
    """The ToF, the pivot to the link, the per-tick counts and (truth arms) the planner's hand on the legs of one rig."""

    def __init__(self, rig: Any, seed: int, arm: str, point: Sequence[float], trace: bool) -> None:
        global _REFLEX_STEP
        from duckmind.neuro import brain as B
        from duckmind.world import tof_model as T

        self.rig = rig
        self.arm = arm
        self.point = (float(point[0]), float(point[1]))
        self.keep_trace = trace
        self.tof = T.TofSensor(rig.duck.model, seed=seed)
        self.frame: Any = None
        self.seen: Any = None
        self.next = 0.0
        self.tracker: Any = None
        self.depth: Any = None
        self.depth_at = -(10**18)
        self.ph: dict[str, float] = {}
        self.prev_t: int | None = None
        self.rx: Any = None
        self.cmd: Any = None
        self.bumps = self.brushes = self.startles = 0
        self.trace: list[list[Any]] = []
        self.events: list[list[Any]] = []
        self.takeovers: list[list[Any]] = []
        self.prev_beh: str | None = None
        self.open_to: list[Any] | None = None
        self.last_touch: float | None = None
        self.on = False  # the click is running (from go_to on)
        self.t0_ms = 0
        self.in_s = 0.0
        self.truth_in_s: float | None = None
        self.near = math.inf
        self.pilot: Any = None
        self.pilot_out: Any = None
        self.piloted = 0
        if _REFLEX_STEP is None:
            _REFLEX_STEP = B.R.step
        rstep = _REFLEX_STEP

        def rstep_w(*a: Any, **k: Any) -> Any:
            st, out = rstep(*a, **k)
            self.rx = out
            return st, out

        B.R.step = rstep_w
        hub, link_tick, brain_step = rig.hub, rig.link.tick, rig.brain.step

        def hub_w(doors: Any) -> Any:
            doors.depth = self.depth_step(doors.state)
            doors.tof = self.frame
            return hub(doors)

        def brain_w(t_ms: int, frame: Any) -> Any:
            out = brain_step(t_ms, frame)
            self.cmd = out[0]
            self.note(t_ms, out[0], frame)
            return out

        def link_w(
            state: Any, request: Any, play: Any, *, depth: Any = None, battery_pct: Any = None
        ) -> Any:
            piv = float(getattr(self.cmd, "pivot", 0.0) or 0.0) if self.cmd is not None else 0.0
            request = replace(request, pivot=piv if float(request.vx) > 0.0 else 0.0)
            if self.on and self.arm in ("truth-reflex", "truth-free"):
                request = self.take_legs(request)
            return link_tick(
                state, request, play, depth=self.depth_fresh(state), battery_pct=battery_pct
            )

        rig.hub = hub_w
        rig.brain.step = brain_w
        rig.link.tick = link_w
        if arm != "main":
            from duckmind.bench import floor as F
            from duckmind.bench import truthplan as TP

            self.pilot = TP.Pilot(F.FloorMap(rig.duck.layout), self.things_at)
            self._thing_ids = self._thing_bodies()
        if arm in ("truth", "truth-notake"):
            orig = rig.brain._errand_drive

            def drive(
                bearing: float,
                range_m: float | None,
                cap: float | None,
                bias: float,
                away: float = 0.0,
            ) -> Any:
                goal = self.believed_goal()
                if goal is None:
                    return orig(bearing, range_m, cap, bias, away=away)
                pose = self.rig.pose()
                self.pilot.update(self.clock(), pose, goal)
                rx = self.rx
                self.piloted += 1
                out = self.pilot.drive(
                    pose,
                    goal,
                    t=self.clock(),
                    cap=None if rx is None else rx.vx_cap,
                    bias=0.0 if rx is None else float(rx.vyaw_bias or 0.0),
                    away=away,
                    can_pivot=bool(self.rig.brain.can_pivot),
                    gate=self.gate(),
                )
                self.pilot_out = (round(out[0], 3), round(out[1], 3), out[2])
                return out[0], out[1], out[2]

            rig.brain._errand_drive = drive

    # ── truth, for the planner and the scoring only ──

    def clock(self) -> float:
        return 0.0 if self.prev_t is None else self.prev_t / 1000.0

    def _thing_bodies(self) -> dict[str, tuple[int, float, float]]:
        """Each movable thing's physics body and where the layout's point of it is from that body's origin."""
        import mujoco

        d = self.rig.duck
        out: dict[str, tuple[int, float, float]] = {}
        for t in d.layout.things:
            b = int(mujoco.mj_name2id(d.model, mujoco.mjtObj.mjOBJ_BODY, t.body))
            if b >= 0:
                x, y = float(d.data.xpos[b][0]), float(d.data.xpos[b][1])
                out[str(t.id)] = (
                    b,
                    float(t.pos_mm[0]) / 1000.0 - x,
                    float(t.pos_mm[1]) / 1000.0 - y,
                )
        return out

    def things_at(self) -> dict[str, tuple[float, float]]:
        xpos = self.rig.duck.data.xpos
        return {
            k: (float(xpos[b][0]) + dx, float(xpos[b][1]) + dy)
            for k, (b, dx, dy) in self._thing_ids.items()
        }

    def gate(self) -> tuple[str, ...]:
        """Why the body link changed the last request (its own gate codes)."""
        last = self.rig.last
        return tuple(getattr(last, "gate", ()) or ()) if last is not None else ()

    def believed_goal(self) -> tuple[float, float] | None:
        """Where the errand's place is in the world by the brain's own belief of it (its odometry frame laid on the
        body's true pose): what its own arrival test will fire on."""
        b = self.rig.brain
        run = b.errand._run
        if run is None:
            return None
        g = compose(self.rig.pose(), compose(inverse(b._odom_now), (run.x, run.y, 0.0)))
        return g[0], g[1]

    def take_legs(self, request: Any) -> Any:
        """(``truth-reflex`` / ``truth-free``) The planner's walk in the place of whatever the brain asked of the legs."""
        rig = self.rig
        if self.truth_in_s is not None or rig.fallen() or not rig.stood:
            return request
        pose = rig.pose()
        self.pilot.update(self.clock(), pose, self.point)
        rx = self.rx if self.arm == "truth-reflex" else None
        self.piloted += 1
        back = False
        if rx is not None and rx.halt:
            vx = vyaw = piv = 0.0
        else:
            vx, vyaw, piv, back = self.pilot.drive(
                pose,
                self.point,
                t=self.clock(),
                cap=None if rx is None else rx.vx_cap,
                bias=0.0 if rx is None else float(rx.vyaw_bias or 0.0),
                can_pivot=bool(rig.brain.can_pivot),
                gate=self.gate(),
                may_back=True,
            )
        self.pilot_out = (round(vx, 3), round(vyaw, 3), piv)
        want = dict(vx=vx, vyaw=vyaw, pivot=piv, posture="stand", approach=False, docking=False, back_arc=back, panic=False,
                    skill=None, skill_manual=False, skill_chain=False, toward=None)  # fmt: skip
        have = {f.name for f in dc_fields(request)}
        return replace(request, **{k: v for k, v in want.items() if k in have})

    # ── the senses the play rig does not carry: the ToF through the depth tracker, as the live driver ──

    def depth_fresh(self, state: Any) -> Any:
        if state.t_ns - self.depth_at > 500_000_000:
            return None
        return self.depth

    def depth_step(self, state: Any) -> Any:
        from duckmind.duckbody import depth as D
        from duckmind.duckbody import protocol as P

        d = self.rig.duck
        now = d.now()
        if now + 1e-9 >= self.next:
            self.next = now + TOF_PERIOD_S
            raw = dict(self.tof.capture(d.world))
            raw["t_ns"] = d.t_ns()
            self.frame = raw
        frame = self.frame
        if frame is None or state.tof is None:
            return self.depth_fresh(state)
        if frame is not self.seen:
            self.seen = frame
            if self.tracker is None:
                self.tracker = D.DepthTracker(standing_height_m=0.12)
            pos, quat = state.tof
            z = float(d.truth()["trunk"][2])
            opos = (float(state.odom_pos[0]), float(state.odom_pos[1]), z)
            pose = D.pose_from_frames(
                pos, quat, state.down or (0.0, 0.0, -1.0), D.trunk_height_mm(opos)
            )
            tof = D.ToFFrame.of(
                int(frame.get("t_ns") or state.t_ns),
                frame.get("distance_mm") or [],
                frame.get("status") or [],
            )
            walking = self.rig.last is not None and self.rig.last.twist != (0.0, 0.0)
            try:
                reading, _ = self.tracker.step(
                    tof,
                    pose,
                    walking=walking,
                    odom=(
                        float(state.odom_pos[0]),
                        float(state.odom_pos[1]),
                        float(state.odom_yaw),
                    ),
                    upright=D.upright(opos),
                )
            except (ValueError, IndexError):
                return None
            if not reading.available:
                self.depth = None
            else:
                far = D.FAR_MM

                def rng(v: Any) -> Any:
                    return None if v >= far else v

                self.depth = P.DepthView(
                    t_ns=state.t_ns,
                    ahead_mm=rng(reading.ahead_mm),
                    left_mm=rng(reading.left_mm),
                    right_mm=rng(reading.right_mm),
                    drop_zones=max(reading.drop_zones, 4)
                    if reading.drop
                    else min(reading.drop_zones, 3),
                    touch=reading.touch,
                    head_pitch_mrad=reading.head_pitch_mrad,
                    ahead_b_mrad=reading.ahead_b_mrad,
                    ahead_top_mm=reading.ahead_top_mm,
                    virtual=reading.virtual,
                    low_edge_mm=reading.low_edge_mm,
                )
            self.depth_at = state.t_ns
        return self.depth_fresh(state)

    # ── the per-tick record ──

    def start(self) -> None:
        """The click begins now (right before ``go_to``)."""
        self.bumps = self.brushes = self.startles = 0
        self.trace.clear()
        self.events.clear()
        self.takeovers.clear()
        self.ph.clear()
        self.open_to = None
        self.last_touch = None
        self.t0_ms = self.prev_t or 0
        self.on = True

    def note(self, t_ms: int, cmd: Any, frame: Any) -> None:
        b = self.rig.brain
        dt = 0.0 if self.prev_t is None else (t_ms - self.prev_t) / 1000.0
        self.prev_t = t_ms
        rx = self.rx
        hits = []
        for e in getattr(rx, "events", ()) or ():
            if e.reflex == "bump_stop":
                self.bumps += 1
                hits.append("bump_stop:" + str(e.stimulus))
            elif e.reflex == "startle":
                self.startles += 1
        for ev in frame.events or ():
            if ev == "blocked":
                self.bumps += 1
                hits.append("blocked")
            elif ev == "brushed":
                self.brushes += 1
                hits.append("brushed")
        if any(h == "bump_stop:tof:touch" for h in hits):
            self.last_touch = t_ms / 1000.0
        if not self.on:
            return
        t = (t_ms - self.t0_ms) / 1000.0
        x, y, yaw = self.rig.pose()
        dist = math.hypot(self.point[0] - x, self.point[1] - y)
        self.near = min(self.near, dist)
        if self.truth_in_s is None:
            self.in_s = self.in_s + dt if dist <= ARRIVE_M else 0.0
            if self.in_s >= ARRIVE_HOLD_S:
                self.truth_in_s = round(t, 2)
        self._takeover(t, b, rx, frame)
        if hits:
            self.events.append(
                [
                    round(t, 2),
                    hits,
                    round(x, 3),
                    round(y, 3),
                    b.behaviour,
                    round(float(cmd.vx), 2),
                    float(cmd.pivot or 0.0),
                    frame.tof.ahead_mm if frame.tof.present else None,
                ]  # fmt: skip
            )
        run = b.errand._run
        if run is None or run.state != "walking":
            if self.keep_trace and self.pilot_out is not None and t_ms % 100 < 20:
                self.trace.append([round(t, 2), round(x, 3), round(y, 3), round(yaw, 3), None, None, None, "no errand", None, None, None,
                                   b.behaviour, self.pilot_out, None if self.rig.last is None else list(self.rig.last.twist)])  # fmt: skip
            return
        if b.clutter.mode == "escape":
            k = "clutter"
        elif b.behaviour != "wander":
            k = "other act"
        elif cmd.vx < -0.01:
            k = "back"
        elif cmd.pivot:
            k = "pivot"
        elif cmd.vx > 0.01:
            k = "walk"
        elif rx is not None and rx.halt:
            k = "stand: halt"
        elif rx is not None and rx.vx_cap is not None and rx.vx_cap <= 0.0:
            k = "stand: cap0"
        elif b.bound.last.cornered and b._back is None:
            k = "stand: cornered"
        elif getattr(b, "_look_stop", None) is not None:
            k = "stand: look"
        else:
            k = "stand: other"
        self.ph[k] = self.ph.get(k, 0.0) + dt
        if self.keep_trace and t_ms % 100 < 20:
            self.trace.append(
                [
                    round(t, 2),
                    round(x, 3),
                    round(y, 3),
                    round(yaw, 3),
                    round(float(cmd.vx), 3),
                    round(float(cmd.vyaw), 3),
                    float(cmd.pivot or 0.0),
                    k,
                    frame.tof.ahead_mm if frame.tof.present else None,
                    None if rx is None else rx.active,
                    round(run.last_range, 3),
                    b.behaviour,
                    self.pilot_out,
                    None if self.rig.last is None else list(self.rig.last.twist),
                ]  # fmt: skip
            )

    def _takeover(self, t: float, b: Any, rx: Any, frame: Any) -> None:
        """An errand that was walking loses the body to another act (the selector's switch this tick), and when it gets
        it back: [t, act, reflex events, the ToF ahead, s since the last touch stop, its length]."""
        run = b.errand._run
        beh = b.behaviour
        prev, self.prev_beh = self.prev_beh, beh
        if self.open_to is not None and (beh == "wander" or run is None):
            self.open_to[-1] = round(t - self.open_to[0], 2)
            self.open_to = None
        if run is None or not run.walk_seen or prev != "wander" or beh == "wander":
            return
        evs = [f"{e.reflex}:{e.stimulus}" for e in getattr(rx, "events", ()) or ()]
        rec = [round(t, 2), beh, evs, frame.tof.ahead_mm if frame.tof.present else None,
               None if self.last_touch is None else round(t + self.t0_ms / 1000.0 - self.last_touch, 2), None]  # fmt: skip
        self.takeovers.append(rec)
        self.open_to = rec


def run_click(job: dict[str, Any]) -> dict[str, Any]:
    """One click of one arm in this process: its row (and its 10 Hz trace under ``_trace`` when asked)."""
    prepare_env()
    from duckmind.bench import knownhome as KH
    from duckmind.neuro.genome import Genome
    from duckmind.world import neuro_play_rig as R

    click, arm = job["click"], job["arm"]
    collision, policies = job["collision"], job["policies"]
    _live_body(collision, policies)
    x, y, yaw = (float(v) for v in click["pose"])
    px, py = (float(v) for v in click["point"])
    seed = int(click["seed"])
    row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
    row.update(schema=SCHEMA, set=click.get("set"), k=int(click["k"]), arm=arm, mode="headless", tree=job.get("tree"), seed=seed,
               pose=[x, y, yaw], point=[px, py], cls=click.get("cls"), line=click.get("line"),
               flagged=bool(click.get("flags")), detour=click.get("detour"))  # fmt: skip
    walk_sha = hashlib.sha256((Path(policies) / "velstand.onnx").read_bytes()).hexdigest()
    genome = Genome.from_seed(seed, gd=50)
    from duckmind.neuro import errand as ER

    global _REASSERT_S
    if _REASSERT_S is None:
        _REASSERT_S = ER.REASSERT_S
    ER.REASSERT_S = 0.0 if arm == "truth-notake" else _REASSERT_S
    with R.PlayNeuro(genome, seed, x=x, y=y, yaw=yaw, home=job["home"], when=WHEN) as rig:
        watch = Watch(rig, seed, arm, (px, py), bool(job.get("trace")))
        link_info = _link_as_live(rig, collision, policies)
        rig.run(SETTLE_S)
        rig.brain.can_pivot = True
        rig.brain.nav_walk = walk_sha
        KH.install(rig.brain, job["known"], rig.pose(), job["home"])
        tx, ty, tyaw = rig.pose()
        rng = math.hypot(px - tx, py - ty)
        br = math.atan2(py - ty, px - tx) - tyaw
        br = math.atan2(math.sin(br), math.cos(br))
        watch.start()
        rig.brain.go_to(br, rng)
        t = 0.0
        arrived_s = None
        free = arm in ("truth-reflex", "truth-free")
        falls, was = 0, False
        while t < CLICK_S:
            rig.run(0.5)
            t += 0.5
            fallen = (
                rig.fallen()
            )  # (sampled twice a second, as every earlier go-there measurement counted falls)
            falls += int(fallen and not was)
            was = fallen
            state = rig.brain.errand.view().get("state")
            if free:
                if watch.truth_in_s is not None:
                    arrived_s = t
                    break
            elif state in ("arrived", "gave_up"):
                if state == "arrived":
                    arrived_s = t
                break
        watch.on = False
        if watch.open_to is not None:
            watch.open_to[-1] = round(t - watch.open_to[0], 2)  # (still another act's at the end)
        p = rig.pose()
        ev = rig.brain.errand.view()
        state = ev.get("state")
        if free:
            state = "arrived" if arrived_s is not None else "walking"
        row.update(
            range_m=round(rng, 2),
            bearing=round(br, 2),
            state=state,
            reason=(ev.get("last") or {}).get("reason") if state == "gave_up" else None,
            arrived=arrived_s is not None,
            arrived_s=arrived_s,
            truth_ok=arrived_s is not None and math.hypot(px - p[0], py - p[1]) <= TRUTH_OK_M,
            truth_in_s=watch.truth_in_s,
            secs=t,
            truth_end=round(math.hypot(px - p[0], py - p[1]), 2),
            near=round(watch.near, 2),
            falls=falls,
            bumps=watch.bumps,
            brushes=watch.brushes,
            startles=watch.startles,
            gave_up=state == "gave_up",
            takeovers=len(watch.takeovers),
            escapes=int(rig.brain._unstick.fired),
            clutter_started=int(rig.brain.clutter.started),
            ph={kk: round(v, 2) for kk, v in sorted(watch.ph.items())},
            rt_min=None,
            takeover_list=watch.takeovers,
            events=watch.events,
            link=link_info,
            walk=walk_sha[:12],
            camera=_camera(),
            plan=None if watch.pilot is None else {**watch.pilot.stats(), "ticks": watch.piloted},
        )
        if job.get("trace"):
            row["_trace"] = watch.trace
    return row


def _camera() -> str | None:
    try:
        from duckmind.world import camera_model as cm

        return (
            str(cm.sim_board())
            if hasattr(cm, "sim_board")
            else os.environ.get("DUCKMIND_BOARD") or getattr(cm, "DEFAULT_SIM_BOARD", None)
        )
    except Exception:  # noqa: BLE001  (an older tree without the board switch)
        return None


# ── the batch ──


def others_running() -> int:
    """How many other test simulators or headless batches run on this machine now (the capacity rule's count; the live
    duck's own server and simulator are not counted, nor this process and its parents)."""
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "simhome|rig_v|duckmind.bench|bench/_boot.py"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    mine = set()
    pid = os.getpid()
    for _ in range(8):
        mine.add(pid)
        try:
            pid = int(
                subprocess.run(
                    ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, timeout=5
                ).stdout.strip()
                or 0
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            break
        if pid <= 1:
            break
    n = 0
    for line in out.splitlines():
        pid_s, _, cmd = line.partition(" ")
        if not pid_s.isdigit() or int(pid_s) in mine:
            continue
        if _is_batch(cmd):
            n += 1
    return n


def _is_batch(cmd: str) -> bool:
    """Whether a process's command line is one test simulator or one headless batch (not the live duck's, not a
    launcher that only wraps one, not a worker of one)."""
    cmd = cmd.strip()
    if "--port 8791" in cmd or "lives/danhuang" in cmd:
        return False  # the live duck's own server and simulator
    if (
        cmd.startswith(("uv ", "nice ", "/bin/sh", "/bin/bash", "bash ", "sh ", "pgrep"))
        or "multiprocessing" in cmd
    ):
        return False
    if "duckmind.world.simhome" in cmd:
        return True  # (a test server is counted once, by its simulator)
    if "rig_v" in cmd and ".py" in cmd:
        return True
    return ("-m duckmind.bench" in cmd or "bench/_boot.py" in cmd) and " compare " not in f"{cmd} "


def gate(
    log: Any = None,
    load_max: float = LOAD_MAX,
    batches_max: int = BATCHES_MAX,
    poll_s: float = 20.0,
) -> None:
    """Wait until the machine has room for one more batch: fewer than ``batches_max`` others and a 1-min load ≤ ``load_max``."""
    while True:
        load, n = os.getloadavg()[0], others_running()
        if load <= load_max and n < batches_max:
            return
        if log is not None:
            log(
                f"waiting for room: load {load:.1f} (max {load_max:g}), {n} other batch(es) (max {batches_max - 1})"
            )
        time.sleep(poll_s)


def default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) // 2 - 1))


def run_batch(
    clicks: Sequence[dict[str, Any]],
    out: Path,
    *,
    arm: str = "main",
    workers: int | None = None,
    known: str | None = None,
    home: str = "real_home",
    collision: str = COLLISION,
    policies: str | None = None,
    tree: str | None = None,
    trace: Path | None = None,
    wait: bool = True,
    log: Any = print,
) -> list[dict[str, Any]]:
    """Run ``clicks`` on ``arm`` and write one row a line to ``out`` (rows already there are kept and not re-run)."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r}: one of {', '.join(ARMS)}")
    prepare_env()
    from duckmind import paths

    policies = str(policies or paths.policy_dir())
    if not (Path(policies) / "velstand.onnx").is_file():
        raise FileNotFoundError(
            f"no walk policy in {policies} (give --policies or set DUCKMIND_POLICY_DIR)"
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict[str, Any]] = {}
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("arm") == arm:
                    done[int(r["k"])] = r
    jobs = [
        {
            "click": c,
            "arm": arm,
            "known": known,
            "home": home,
            "collision": collision,
            "policies": policies,
            "tree": tree,
            "trace": trace is not None,
        }
        for c in clicks
        if int(c["k"]) not in done
    ]
    if jobs and wait:
        gate(log)
    t0 = time.time()
    tr = gzip.open(trace, "at") if trace is not None else None
    with out.open("a", encoding="utf-8") as f:
        if jobs:
            with ProcessPoolExecutor(max_workers=workers or default_workers()) as ex:
                for n, row in enumerate(ex.map(run_click, jobs), 1):
                    rtrace = row.pop("_trace", None)
                    if tr is not None:
                        tr.write(json.dumps({"k": row["k"], "arm": arm, "trace": rtrace}) + "\n")
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    done[int(row["k"])] = row
                    if log is not None and (n % 20 == 0 or n == len(jobs)):
                        got = sum(1 for r in done.values() if r["arrived"])
                        log(
                            f"[{arm}] {n}/{len(jobs)} clicks, arrived {got}/{len(done)}, {time.time() - t0:.0f} s"
                        )
    if tr is not None:
        tr.close()
    return [done[int(c["k"])] for c in clicks if int(c["k"]) in done]


def load_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main_script() -> list[str]:
    """How to start this bench in another interpreter (``sys.executable`` + these): through ``_boot.py``, so that the
    tree on ``PYTHONPATH`` is the one measured and the bench is this one."""
    return [sys.executable, str(Path(__file__).resolve().parent / "_boot.py")]


__all__ = [
    "ARMS",
    "CLICK_S",
    "ROW_FIELDS",
    "SCHEMA",
    "gate",
    "load_rows",
    "main_script",
    "others_running",
    "run_batch",
    "run_click",
]
