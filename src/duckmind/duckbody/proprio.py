"""Body sense (spec §5.2): what the duck feels from ``robot.state`` alone.

One :class:`Proprio` per duck, stepped once per ``robot.state`` tick (clock = the tick's own
``t_ns``, so the board, the live simulator, the headless rig and replays run the same code). It
keeps *levels* (:class:`~duckmind.duckbody.protocol.ProprioView`) and reports *edges*:

``lifted`` / ``set_down`` — picked up / put down (owner B3, B4). Two stages, because a duck
    whose walk command keeps streaming steps in the air with its legs as loaded as on the floor
    (phase-3 lift trials: leg load ≈ 130 held vs ≈ 110 walking), so load alone cannot see it:

    1. ``unloaded`` (*unsure*, no claim): the legs went light (load under ``unsure_load`` for
       ``unsure_hold_s``; during a reverse's first ``reverse_spinup_s`` under
       ``unsure_load_reverse``: the reverse start's own dip reaches 20.4–23.9, a held duck's
       legs carry 9–13), or a walk is commanded but the trunk does not rock the way a stepping
       duck on a floor does (gyro RMS under ``quiet_gyro`` for ``quiet_hold_s``). The body loop
       stops the walk (reason ``reflex_unloaded``) and says nothing about being picked up.
    2. ``lifted`` (the claim): with the walk stopped, the leg load stays under ``lift_load`` for
       ``lift_hold_s`` (a held duck's legs carry nothing: ≈ 9 standing-held, ≈ 13 seated-held,
       against ≈ 35 standing and ≈ 25 seated on the floor) — or the legs keep flailing in the
       air after the stop (0.5 s mean load over ``flail_load`` with the trunk still, stopped
       ≥ 1 s: velstand at zero command cannot find the floor). The body loop then sits (B4).

    An ``unloaded`` that the floor explains (load back above ``clear_load`` for ``clear_hold_s``
    with the walk stopped) ends with ``unloaded_clear``. ``set_down`` is the load coming back
    (``down_load`` for ``down_hold_s``) while lifted. Both lift edges start a new odometry epoch:
    contact odometry breaks while carried.

    The thresholds are the simulator's leg-force proxy (``currents_ma`` = |actuator force| ×
    100, uncalibrated): owner B3 says use them now and recalibrate on the board.

``fell`` / ``up`` — one fall episode (spec §5.4, owner B1): trigger A = robotd's
    ``safety.fallen`` (rising, or still up when nothing holds the duck); trigger B = tilt over 40° held 1 s while robotd still says
    upright (a duck stuck nose-down on a threshold at 43–60° is never ``fallen`` to robotd). The episode
    ends when the duck has been up (tilt < 25°, not fallen, policy walk/stand) for 1 s, so a
    flag that flickers is one fall. A duck already claimed ``lifted`` does not fall: held tilted
    in someone's arms (a cuddle) keeps the lift claim, and nothing tries to right it in the hand;
    it is on the floor again only at ``set_down``. A fall the claim held back is not lost: the claim above 35° of tilt cannot
    set down (the floor bands are off), so when the body link ends it (its floor re-test found the legs carry the trunk) and robotd
    still says fallen, the fall begins at that tick (docs/design/neural-duck/LIFT-LOCK-RIGHTING-2026-10-18.md).

``blocked`` / ``brushed`` / ``stumbled`` — contacts (spec §5.2; phase 1 ``obstacles.md``):
    *blocked* = progress along the walk under 30 % of what the gait makes for 0.6 s with a heading
    jerk, ToF touch or doubled leg load to back it, or for 2 s on odometry alone (marching on the
    spot counts); *brushed* (grazed) = a 15–60° heading swing the command does not explain while
    still progressing — "explain" by the gait's measured turn (bias + gain × vyaw, the
    motion model walkexec's look_veto uses: a straight walk drifts ~10°/s right and an arc turns
    at ~60 % of its command), not by the command itself; *stumbled* (tripped) = trunk rate over 3 rad/s with doubled leg load, a
    fall flag or a tilt over 20°. Odometry that never moves (the headless path) disables blocked.

Pure: no clock, no I/O; integers where they leave the module (:class:`ProprioView`).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from duckmind.duckbody.footfalls import Footfalls
from duckmind.duckbody.protocol import ProprioView
from duckmind.duckbody.robotd_client import RobotState

NS = 1_000_000_000
UP_POLICIES = frozenset({"walk", "stand"})
POSTURE_POLICIES = frozenset({"walk", "stand", "sit"})


@dataclass(frozen=True)
class ProprioConfig:
    # ── lifted (owner B3: the sim proxy now, the board later) ──
    unsure_load: float = 24.0
    unsure_load_sit: float = 20.5
    unsure_hold_s: float = 0.15
    quiet_gyro: float = 0.15  # rad/s RMS over quiet_window_s while a walk is commanded
    quiet_window_s: float = 0.5
    quiet_hold_s: float = 0.3
    quiet_spinup_s: float = 1.0  # a new walk rocks the trunk only after its first steps
    # a reverse spins up in ~0.9–1.2 s, its leg load sliding to 20.4–23.9 around 0.45 s before
    # the first backward step (2026-09-28 rig: 48 open-floor starts, the cornered back steps;
    # 9 of 24 of those read "unsure" at 24): during a reverse's spin-up only a load a held duck
    # has (≈ 9–13) counts as light, and the trunk may stay quiet a little longer
    unsure_load_reverse: float = 18.0
    reverse_spinup_s: float = 1.2
    quiet_spinup_reverse_s: float = 1.5
    walk_vx: float = 0.25  # |vx| at or above this counts as a commanded walk
    lift_load: float = 20.0
    lift_load_sit: float = 18.5
    lift_hold_s: float = 0.30
    lift_hold_sit_s: float = 0.40
    flail_load: float = 100.0  # mean over quiet_window_s: stopped, trunk still, legs working hard
    flail_stopped_s: float = 1.0
    flail_hold_s: float = 0.5
    # the floor bands: a duck standing / sitting on a floor, stopped, loads its legs like this
    floor_band: tuple[float, float] = (28.0, 120.0)
    floor_band_sit: tuple[float, float] = (21.5, 70.0)
    clear_hold_s: float = 0.5
    down_hold_s: float = 0.6
    down_calm_gyro: float = 0.3  # rad/s RMS: the set-down needs a trunk that is not carried
    settle_after_policy_s: float = 1.5  # a posture change swings the legs: ignore its load
    lift_max_tilt_deg: float = 35.0
    # ── falls (spec §5.4, owner B1) ──
    fall_tilt_deg: float = 40.0
    fall_tilt_hold_s: float = 1.0
    up_tilt_deg: float = 25.0
    up_hold_s: float = 1.0
    # ── contacts (spec §5.2) ──
    blocked_ratio: float = 0.30
    blocked_window_s: float = 0.8
    blocked_confirm_s: float = 0.6
    blocked_alone_s: float = 2.0
    blocked_spinup_s: float = 1.0
    jerk_rad: float = 0.15
    brush_min_rad: float = math.radians(15.0)
    brush_max_rad: float = math.radians(60.0)
    brush_window_s: float = 1.0
    # the gait's own turn (rad/s = bias + gain × vyaw): phase 1 vx 0.40 (lookplan's motion model;
    # the phase-3 A3 medians at 0.40–0.55 lie within 6°/s of it); reversing (dockexec −0.40) turns
    # −4.5°/s
    yaw_bias: float = -0.175
    yaw_gain: float = 0.597
    yaw_bias_reverse: float = -0.079
    stumble_gyro: float = 3.0
    stumble_load_x: float = 2.0
    stumble_tilt_deg: float = 20.0
    refractory_s: float = 1.5
    # robotd's quick turn on the spot (quick_turn_*) and its hand-back: the spin and the loaded legs are the
    # turn itself for this long after it (live 2026-10-24: 22 of 29 'stumbled' edges at a turn's end); only
    # a trunk that tips (or a fall) is a trip then
    own_turn_s: float = 0.6

    def expected_yaw_rate(self, vx: float, vyaw: float) -> float:
        """How fast the walk turns at this command (rad/s): the gait's measured model."""
        if vx == 0.0:
            return vyaw
        return (self.yaw_bias if vx > 0 else self.yaw_bias_reverse) + self.yaw_gain * vyaw

    def expected_speed(self, vx: float) -> float:
        """Path speed the gait makes at this command (motion_model.json speed_model: 0.163 /
        0.178 / 0.195 m/s at 0.30 / 0.35 / 0.40, extended linearly)."""
        return max(0.0, 0.067 + 0.32 * abs(vx))


class _Held:
    """How long a condition has held, on the tick clock."""

    def __init__(self) -> None:
        self.since: int | None = None

    def update(self, on: bool, t_ns: int) -> float:
        if not on:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = t_ns
        return (t_ns - self.since) / NS


class _Window:
    """(t_ns, value) samples over the last ``span_s`` seconds."""

    def __init__(self, span_s: float) -> None:
        self.span = int(span_s * NS)
        self.items: deque[tuple[int, float]] = deque()

    def add(self, t_ns: int, value: float) -> None:
        self.items.append((t_ns, value))
        while self.items and t_ns - self.items[0][0] > self.span:
            self.items.popleft()

    def covered_s(self) -> float:
        if len(self.items) < 2:
            return 0.0
        return (self.items[-1][0] - self.items[0][0]) / NS

    def rms(self) -> float:
        if not self.items:
            return 0.0
        return math.sqrt(sum(v * v for _, v in self.items) / len(self.items))

    def mean(self) -> float:
        if not self.items:
            return 0.0
        return sum(v for _, v in self.items) / len(self.items)

    def median(self) -> float:
        values = sorted(v for _, v in self.items)
        if not values:
            return 0.0
        return values[len(values) // 2]


@dataclass(frozen=True)
class ProprioOut:
    view: ProprioView
    edges: tuple[str, ...] = ()
    fell_trigger: str | None = None  # "fallen" (A) or "tilt40" (B) on a ``fell`` edge
    contact_side: str = "unknown"
    evidence: tuple[str, ...] = ()
    lift_conf_ppm: int = 0
    load: float | None = None


@dataclass
class _Walk:
    start_ns: int | None = None
    vx: float = 0.0
    vyaw: float = 0.0


@dataclass
class Proprio:
    cfg: ProprioConfig = field(default_factory=ProprioConfig)
    feet: Footfalls | None = None  # the step counter (needs the leg model); None → steps stay 0

    def __post_init__(self) -> None:
        c = self.cfg
        self.lift = "ground"  # ground / unsure / lifted
        self.why_unsure = ""
        self.epoch = 0
        self.fall_open = False
        self.prev_fallen = False
        self.odom_seen_moving = False
        self.walk = _Walk()
        self.gyro_win = _Window(c.quiet_window_s)
        self.load_win = _Window(2.0)
        self.load_short = _Window(c.quiet_window_s)
        self.odom_win: deque[tuple[int, float, float, float]] = deque()
        self._quiet = _Held()
        self._light = _Held()
        self._lift = _Held()
        self._clear = _Held()
        self._down = _Held()
        self._flail = _Held()
        self._stopped = _Held()
        self._policy = ""
        self._policy_since = 0
        self._turn_ns = -(10**18)  # the last tick robotd ran its quick turn
        self._tilt = _Held()
        self._up = _Held()
        self._slow = _Held()
        self._last_edge: dict[str, int] = {}
        self._flicker = 0
        self.steps = 0

    def drop_claim(self) -> None:
        """The body link judged the claim false (the legs carry the trunk: a floor or a ledge bears the duck, no hand holds it): no
        claim stands and every hold that built one starts over. The body link says the ``set_down`` edge and calls this every tick
        the claim is to stay down; the tick it stops, light legs claim again after their own holds (0.15 s, then 0.4 s)."""
        if self.lift == "lifted":
            self.epoch += 1  # (as a set-down: the odometry frame the claim began breaks again)
        self.lift = "ground"
        for held in (self._light, self._lift, self._clear, self._down, self._flail):
            held.update(False, 0)

    # ── helpers ──

    def _posture(self, policy: str) -> str:
        if policy == "sit":
            return "sit"
        if policy in UP_POLICIES:
            return "stand"
        return "unknown"

    def _refractory(self, name: str, t_ns: int) -> bool:
        last = self._last_edge.get(name)
        if last is not None and t_ns - last < int(self.cfg.refractory_s * NS):
            return True
        self._last_edge[name] = t_ns
        return False

    # ── the step ──

    def step(
        self,
        state: RobotState,
        *,
        cmd: tuple[float, float] | None = None,
        touch: bool = False,
    ) -> ProprioOut:
        c = self.cfg
        t = state.t_ns
        edges: list[str] = []
        evidence: list[str] = []
        vx, vyaw = cmd if cmd is not None else (state.requested[0], state.requested[2])
        walking_cmd = abs(vx) >= c.walk_vx
        if walking_cmd:
            if self.walk.start_ns is None or abs(vx - self.walk.vx) > 0.05:
                self.walk = _Walk(t, vx, vyaw)
            self.walk.vyaw = vyaw
        else:
            self.walk = _Walk()
        walk_age = 0.0 if self.walk.start_ns is None else (t - self.walk.start_ns) / NS
        tilt = state.tilt_rad
        tilt_deg = math.degrees(tilt) if tilt is not None else 0.0
        load = state.leg_load
        gyro_mag = None
        if state.gyro is not None:
            g = state.gyro
            gyro_mag = math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
            self.gyro_win.add(t, gyro_mag)
        base_load = self.load_win.median() if self.load_win.items else None
        if load is not None:
            self.load_win.add(t, load)
            self.load_short.add(t, load)
        posture = self._posture(state.policy)
        if self.feet is not None:
            self.feet.update(state)
            self.steps = self.feet.steps

        # ── falls: trigger A / B, one episode until up for 1 s ──
        fell_trigger = None
        rose = state.fallen and not self.prev_fallen
        self.prev_fallen = state.fallen
        if rose:
            self._flicker += 1
        stuck_s = self._tilt.update(
            (not state.fallen) and tilt is not None and tilt_deg > c.fall_tilt_deg, t
        )
        if not self.fall_open:
            if state.fallen:
                fell_trigger = "fallen"
            elif stuck_s >= c.fall_tilt_hold_s - 1e-9:
                fell_trigger = "tilt40"
            if fell_trigger is not None and self.lift == "lifted":
                fell_trigger = None  # held in the arms: tilting there is not a fall
            if fell_trigger is not None:
                self.fall_open = True
                edges.append("fell")
                self._up.update(False, t)
        else:
            up = (
                not state.fallen
                and tilt is not None
                and tilt_deg < c.up_tilt_deg
                and state.policy in UP_POLICIES
            )
            if self._up.update(up, t) >= c.up_hold_s - 1e-9:
                self.fall_open = False
                self._flicker = 0
                edges.append("up")

        # ── lifted: unsure → lifted → set down ──
        if state.policy != self._policy:
            self._policy, self._policy_since = state.policy, t
        settled = (t - self._policy_since) / NS >= c.settle_after_policy_s
        lift_ok = (
            load is not None
            and not self.fall_open
            and state.policy in POSTURE_POLICIES
            and tilt_deg < c.lift_max_tilt_deg
        )
        sit = posture == "sit"
        conf = 0
        stopped = not walking_cmd
        stopped_s = self._stopped.update(stopped, t)
        if self.fall_open and self.lift != "ground":
            self.lift = "ground"  # a fall supersedes an unsure/lifted guess
        if lift_ok and load is not None:
            unsure_load = c.unsure_load_sit if sit else c.unsure_load
            reversing = walking_cmd and vx < 0
            if reversing and walk_age < c.reverse_spinup_s:
                unsure_load = min(unsure_load, c.unsure_load_reverse)
            lift_load = c.lift_load_sit if sit else c.lift_load
            lift_hold = c.lift_hold_sit_s if sit else c.lift_hold_s
            lo, hi = c.floor_band_sit if sit else c.floor_band
            on_floor_load = lo <= load <= hi and stopped and settled
            trunk_still = (
                gyro_mag is not None
                and self.gyro_win.covered_s() >= c.quiet_window_s * 0.8
                and self.gyro_win.rms() < c.quiet_gyro
            )
            light_s = self._light.update(load < unsure_load, t)
            quiet_s = self._quiet.update(
                trunk_still
                and state.policy == "walk"
                and walking_cmd
                and walk_age >= (c.quiet_spinup_reverse_s if reversing else c.quiet_spinup_s),
                t,
            )
            lift_s = self._lift.update(load < lift_load and stopped, t)
            flail_s = self._flail.update(
                trunk_still
                and stopped
                and stopped_s >= c.flail_stopped_s
                and state.policy in UP_POLICIES
                and self.load_short.covered_s() >= c.quiet_window_s * 0.8
                and self.load_short.mean() > c.flail_load,
                t,
            )
            if self.lift == "ground":
                if (
                    light_s >= c.unsure_hold_s - 1e-9
                    or quiet_s >= c.quiet_hold_s - 1e-9
                    or flail_s > 0.0
                ):
                    self.lift = "unsure"
                    self.why_unsure = (
                        "light" if light_s > 0 else ("quiet" if quiet_s > 0 else "flail")
                    )
                    edges.append("unloaded")
                    self._clear.update(False, t)
            if self.lift == "unsure":
                clear_s = self._clear.update(on_floor_load, t)
                if lift_s >= lift_hold - 1e-9 or flail_s >= c.flail_hold_s - 1e-9:
                    self.lift = "lifted"
                    self.epoch += 1
                    edges.append("lifted")
                    self._down.update(False, t)
                    if flail_s >= c.flail_hold_s - 1e-9:
                        conf = 700_000
                    else:
                        conf = 900_000 if load < 0.6 * lift_load else 750_000
                elif clear_s >= c.clear_hold_s - 1e-9:
                    self.lift = "ground"
                    edges.append("unloaded_clear")
            elif self.lift == "lifted":
                # on the floor: a floor-like load *and* a trunk no hand is swinging (the carry
                # to the set-down spot swings it at 1–4 rad/s while the legs load up)
                calm = (
                    gyro_mag is not None
                    and self.gyro_win.covered_s() >= c.quiet_window_s * 0.8
                    and self.gyro_win.rms() < c.down_calm_gyro
                )
                if self._down.update(on_floor_load and calm, t) >= c.down_hold_s - 1e-9:
                    self.lift = "ground"
                    self.epoch += 1
                    edges.append("set_down")
                    conf = 800_000
        else:
            self._light.update(False, t)
            self._quiet.update(False, t)
            self._lift.update(False, t)
            self._flail.update(False, t)
            self._clear.update(False, t)
            self._down.update(False, t)

        # ── contacts (only on the floor, not in a fall) ──
        blocked = brushed = stumbled = False
        side = "unknown"
        x, y, _ = state.odom_pos
        yaw = state.odom_yaw
        if (x, y, yaw) != (0.0, 0.0, 0.0):
            self.odom_seen_moving = True
        self.odom_win.append((t, x, y, yaw))
        while self.odom_win and t - self.odom_win[0][0] > int(
            max(c.blocked_window_s, c.brush_window_s) * NS
        ):
            self.odom_win.popleft()
        on_floor = self.lift == "ground" and not self.fall_open
        if state.policy.startswith("quick_turn"):
            self._turn_ns = t
        own_turn = t - self._turn_ns <= int(c.own_turn_s * NS)
        if on_floor and gyro_mag is not None and gyro_mag > c.stumble_gyro:
            backed = []
            if load is not None and base_load is not None and base_load > 1.0 and not own_turn:
                if load >= c.stumble_load_x * base_load:
                    backed.append("load")
            if state.fallen:
                backed.append("fallen")
            if tilt_deg > c.stumble_tilt_deg:
                backed.append("tilt")
            if backed and not self._refractory("stumbled", t):
                stumbled = True
                edges.append("stumbled")
                evidence += ["gyro", *backed]
        # (not while robotd runs its own quick turn on the spot, or just after: the odometry is not meant to go forward then and
        # the heading turns on purpose — a turn once round gave a 'blocked' 1.5-1.9 s into it every time, FlipRoom 2026-10-27)
        if (
            on_floor
            and walking_cmd
            and not own_turn
            and self.odom_seen_moving
            and walk_age >= c.blocked_spinup_s
            and len(self.odom_win) >= 2
        ):
            t0, x0, y0, yaw0 = self.odom_win[0]
            span = (t - t0) / NS
            if span >= c.blocked_window_s * 0.8:
                path = math.hypot(x - x0, y - y0)
                want = c.expected_speed(vx) * span
                slow = path < c.blocked_ratio * want
                slow_s = self._slow.update(slow, t)
                dyaw = _wrap(yaw - yaw0)
                expect_yaw = c.expected_yaw_rate(vx, vyaw) * span
                jerk = abs(_wrap(dyaw - expect_yaw)) > c.jerk_rad
                heavy = (
                    load is not None
                    and base_load is not None
                    and base_load > 1.0
                    and load >= c.stumble_load_x * base_load
                )
                backed = [
                    n for n, on in (("heading", jerk), ("touch", touch), ("load", heavy)) if on
                ]
                if slow and (
                    (backed and slow_s >= c.blocked_confirm_s - 1e-9)
                    or slow_s >= c.blocked_alone_s - 1e-9
                ):
                    if not self._refractory("blocked", t):
                        blocked = True
                        edges.append("blocked")
                        evidence += ["odom", *backed]
                        side = (
                            "front" if not jerk else ("left" if dyaw - expect_yaw < 0 else "right")
                        )
                elif not slow and span >= c.brush_window_s * 0.8:
                    swing = _wrap(dyaw - expect_yaw)
                    if c.brush_min_rad <= abs(swing) <= c.brush_max_rad:
                        if not self._refractory("brushed", t):
                            brushed = True
                            edges.append("brushed")
                            evidence += ["heading"]
                            # turned away from the side that was touched
                            side = "right" if swing > 0 else "left"
        else:
            self._slow.update(False, t)

        view = ProprioView(
            posture=posture,
            policy=state.policy,
            lifted=self.lift == "lifted",
            unloaded=self.lift == "unsure",
            fallen=self.fall_open,
            tilt_mrad=int(round((tilt or 0.0) * 1000)),
            blocked=blocked,
            brushed=brushed,
            stumbled=stumbled,
            steps=self.steps,
            odom_epoch=self.epoch,
        )
        return ProprioOut(
            view=view,
            edges=tuple(edges),
            fell_trigger=fell_trigger,
            contact_side=side,
            evidence=tuple(evidence),
            lift_conf_ppm=conf,
            load=load,
        )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["Proprio", "ProprioConfig", "ProprioOut"]
