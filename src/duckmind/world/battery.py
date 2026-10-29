"""The world's battery: a pack that drains with what the body really does and charges on the
nest's contact (physical-design spec §10.7; owner decisions E1, E2).

**Display only in phase 2 (E1).** The world knows how full the pack is; robotd does not: the
official ``slow`` op keeps reporting the nominal 7.4 V (simhome refuses to replace it), because
robotd scales every action by 7.4/V and, on an empty pack, sits the duck down, cuts its torque
and tries to power the machine off — and before phase 5 the duck never walks onto its nest by
itself, so a fed battery would only run flat. The page shows this number next to Duck Core's
own. **Low-voltage weakness (E2)** is a model parameter only: :func:`motor_scale_ppm` says how
much weaker the servos would be at this voltage; nothing applies it until phase 5.

:class:`Pack` is pure integer arithmetic (mA, mV, ms, parts per million), deterministic, with
no MuJoCo, for HomeWorld later:

* the pack is upstream's NP-F550 2S Li-ion (``duck-control`` ``model.rs``: 8.2 V off a full
  charge under load, 6.6 V the sag floor), :data:`CAPACITY_MAH` mAh; ``health_ppm`` shrinks the
  capacity and raises the resistance as the pack ages;
* open-circuit voltage from a 2S Li-ion table (:data:`OCV_TABLE`), minus the sag of the load
  through the pack's resistance (:data:`R_PACK_MOHM`);
* the load is the board (:data:`BOARD_MA`: Radxa, camera, microphone) plus each servo's idle
  draw and a draw proportional to its torque (:data:`SERVO_MA_PER_NM`: XL330-M288 stall 1.47 A
  at 0.52 N·m); the body server's own ``currents_ma`` (|torque| × 100) is a stand-in and not
  used. With the servo torques a policy-driven duck really has (phase-2 headless measurement:
  20 / 31 / 75 mN·m per servo seated / standing / walking) the pack drains 1.56 / 1.87 /
  3.2 %/min, next to v4's 1.6 %/min seated, ×1.3 standing, ×2 walking (~62 min a pack);
  tests/test_world_battery.py checks it;
* charging only while the nest's contact is closed (the PG pin): a power path feeds the body
  from the charger and charges the pack at :data:`CHARGE_MA` (1.25 %/min net, v4's hardware
  requirement), tapering above :data:`TAPER_FROM_PPM` to a full stop.

The world part (:class:`WorldBattery`) integrates the load once per robotd tick of physics
time (every 20 ms) from the duck's actuator torques.

Ops (evaluator/world)::

    {"op": "x_battery", "do": "status" | "swap" | "set" | "power" | "load" | "health",
     "soc_ppm": int, "on": bool, "name": str, "ma": int, "health_ppm": int}

``x_world`` → ``battery``: ``{"model": "world", "fed_to_robotd": false, "soc_ppm", "percent",
"volts_mv", "ocv_mv", "current_ma", "load_ma", "charge_ma", "charging", "pg", "powered",
"robotd_percent_if_fed", "motor_scale_ppm", "weakness", "health_ppm", "capacity_mah",
"empty", "drain_ppm_per_min"}``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from duckmind.world.worldkit import RobotIds, register_part

PPM = 1_000_000
MS_PER_HOUR = 3_600_000
CAPACITY_MAH = 2200  # Sony NP-F550 (upstream duck-control: "NP-F550, 2S Li-ion")
R_PACK_MOHM = 120  # pack + wiring: 8.4 V full → ~8.2 V under a 2 A load (upstream's span)
# 2S open-circuit voltage by state of charge (ppm → mV), a typical Li-ion curve
OCV_TABLE: tuple[tuple[int, int], ...] = (
    (0, 6600),
    (30_000, 6900),
    (60_000, 7100),
    (100_000, 7200),
    (200_000, 7320),
    (300_000, 7420),
    (400_000, 7500),
    (500_000, 7580),
    (600_000, 7680),
    (700_000, 7800),
    (800_000, 7960),
    (900_000, 8160),
    (1_000_000, 8400),
)
FULL_MV = 8200  # upstream BATTERY_FULL_V: "off a full charge, under load"
EMPTY_MV = 6600  # upstream BATTERY_EMPTY_V: "the sag floor"
NOMINAL_MV = 7400  # body_server.NOMINAL_VOLTS, what robotd is told in phase 2
BOARD_MA = 520  # the board, camera and microphone, seen at the pack
SERVO_IDLE_MA = 55  # one servo, torque on, holding nothing
SERVO_OFF_MA = 15  # one servo, torque off
SERVO_MA_PER_NM = 2800  # XL330-M288: stall 1.47 A at 0.52 N·m
QUIESCENT_MA = 1  # everything off
CHARGE_MA = 1650  # 0.75 C: 1.25 %/min net with a power path (v4's hardware requirement)
TAPER_FROM_PPM = 850_000  # constant current up to here, then tapering to full
TAPER_MIN_MA = 60
CHARGE_MAX_MV = 8400
EMPTY_PPM = 0
EVERY_MS = 20  # the world part integrates once per robotd tick of physics time


def ocv_mv(soc_ppm: int) -> int:
    """Open-circuit voltage (mV) of the 2S pack at a state of charge (ppm), by the table."""
    soc = max(0, min(PPM, soc_ppm))
    for (s0, v0), (s1, v1) in zip(OCV_TABLE, OCV_TABLE[1:], strict=False):
        if soc <= s1:
            return v0 + (v1 - v0) * (soc - s0) // (s1 - s0)
    return OCV_TABLE[-1][1]


def servo_ma(torque_mnm: int, *, torque_on: bool = True) -> int:
    """One servo's draw (mA) at a torque (mN·m, absolute)."""
    if not torque_on:
        return SERVO_OFF_MA
    return SERVO_IDLE_MA + SERVO_MA_PER_NM * abs(torque_mnm) // 1000


def robotd_percent(volts_mv: int) -> int:
    """What robotd would show for a bus voltage (upstream ``battery_percent``: linear
    6.6→8.2 V), 0–100."""
    if volts_mv <= 0:
        return 0
    value = (volts_mv - EMPTY_MV) * 100 // (FULL_MV - EMPTY_MV)
    return max(0, min(100, value))


def motor_scale_ppm(volts_mv: int, *, weakness: bool = False) -> int:
    """E2: how strong the servos are at this voltage (torque ∝ supply voltage, relative to the
    7.4 V robotd assumes), in ppm. With ``weakness`` off (phase 2) it is always full strength;
    the would-be value is still reported so it can be measured."""
    if not weakness:
        return PPM
    return max(0, min(PPM, volts_mv * PPM // NOMINAL_MV))


def would_be_scale_ppm(volts_mv: int) -> int:
    return motor_scale_ppm(volts_mv, weakness=True)


@dataclass
class Pack:
    """The battery pack (integers only; see the module docstring)."""

    capacity_mah: int = CAPACITY_MAH
    health_ppm: int = PPM  # 1e6 new; lower is older (less capacity, more resistance)
    soc_ppm: int = PPM
    r_mohm: int = R_PACK_MOHM
    charge_ma: int = CHARGE_MA
    _rest_mams: int = 0  # charge below one ppm step, carried over (mA·ms)

    @property
    def capacity_mams(self) -> int:
        """Usable charge in mA·ms at this health."""
        return self.capacity_mah * MS_PER_HOUR * self.health_ppm // PPM

    @property
    def resistance_mohm(self) -> int:
        """Older packs sag more: the resistance grows as health falls."""
        return self.r_mohm * PPM // max(1, self.health_ppm)

    def charging_ma(self, pg: bool) -> int:
        """The charger's current into the pack (0 without power on the contact)."""
        if not pg or self.soc_ppm >= PPM:
            return 0
        if self.soc_ppm < TAPER_FROM_PPM:
            return self.charge_ma
        left = PPM - self.soc_ppm
        return max(TAPER_MIN_MA, self.charge_ma * left // (PPM - TAPER_FROM_PPM))

    def current_ma(self, load_ma: int, pg: bool) -> int:
        """The pack's current: + discharging, − charging (with a power path the charger feeds
        the load, so on the contact the pack only charges)."""
        if pg:
            return -self.charging_ma(True)
        return max(0, load_ma)

    def volts_mv(self, load_ma: int, pg: bool = False) -> int:
        """The pack's voltage under this load (what the servos would read as their supply)."""
        current = self.current_ma(load_ma, pg)
        volts = ocv_mv(self.soc_ppm) - current * self.resistance_mohm // 1000
        return min(CHARGE_MAX_MV, max(0, volts))

    def step(self, dt_ms: int, load_ma: int, pg: bool = False) -> int:
        """Run ``dt_ms`` at this load; returns the pack current (mA)."""
        current = self.current_ma(load_ma, pg)
        cap = self.capacity_mams
        if cap <= 0 or dt_ms <= 0:
            return current
        charge = self.soc_ppm * cap // PPM + self._rest_mams - current * dt_ms
        charge = max(0, min(cap, charge))
        self.soc_ppm = charge * PPM // cap
        self._rest_mams = charge - self.soc_ppm * cap // PPM
        return current

    def swap(self, soc_ppm: int = PPM) -> None:
        """A different pack (a fresh one by default)."""
        self.soc_ppm = max(0, min(PPM, soc_ppm))
        self._rest_mams = 0

    @property
    def empty(self) -> bool:
        return self.soc_ppm <= EMPTY_PPM


def drain_ppm_per_min(load_ma: int, capacity_mah: int = CAPACITY_MAH) -> int:
    """How fast a steady load empties a pack (ppm of capacity per minute)."""
    return load_ma * 60_000 * PPM // (capacity_mah * MS_PER_HOUR)


def body_load_ma(torques_mnm: Iterable[int], *, torque_on: bool, board_on: bool = True) -> int:
    """The body's draw: the board plus every servo."""
    if not board_on:
        return QUIESCENT_MA
    return BOARD_MA + sum(servo_ma(t, torque_on=torque_on) for t in torques_mnm)


# ── the world part ──────────────────────────────────────────────────────────────


@dataclass
class WorldBattery:
    """The pack in the physics world (display only: see the module docstring)."""

    contact: Callable[[], bool] | None = None  # the nest's PG pin, when there is a nest
    pack: Pack = field(default_factory=Pack)
    weakness: bool = False  # E2: off in phase 2 (measure only)
    name: str = "battery"
    powered: bool = True
    loads: dict[str, int] = field(default_factory=dict)  # extra named loads (mA)
    load_ma: int = BOARD_MA
    current: int = 0
    pg: bool = False
    _actuators: list[int] = field(default_factory=list)
    _body: Any = None
    _next_ms: int = 0
    _last_ms: int | None = None
    _history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=64))

    def edit_spec(self, spec: Any, home: Any) -> None:
        return None

    def ready(self, world: Any, body: Any = None) -> None:
        model = world.model
        robot = RobotIds.of(model)
        self._body = body
        self._actuators = [
            a
            for a in range(int(model.nu))
            if int(model.jnt_bodyid[int(model.actuator_trnid[a][0])]) in robot.bodies
        ]
        self._last_ms = int(round(float(world.data.time) * 1000))
        self._next_ms = self._last_ms

    def after_step(self, world: Any) -> None:
        now = int(round(float(world.data.time) * 1000))
        if now < self._next_ms:
            return
        self._next_ms = now + EVERY_MS
        last = now if self._last_ms is None else self._last_ms
        self._last_ms = now
        self.integrate(world.data, now - last)

    def integrate(self, data: Any, dt_ms: int) -> None:
        torque_on = bool(getattr(self._body, "torque_on", True)) if self._body is not None else True
        torques = [int(round(abs(float(data.actuator_force[a])) * 1000)) for a in self._actuators]
        load = body_load_ma(torques, torque_on=torque_on, board_on=self.powered)
        if self.powered:
            load += sum(self.loads.values())
        self.load_ma = load
        self.pg = bool(self.contact()) if self.contact is not None else False
        self.current = self.pack.step(dt_ms, load, self.pg)
        self._history.append((int(round(float(data.time) * 1000)), self.pack.soc_ppm))

    def drain_per_min(self) -> int | None:
        """The recent drain (ppm of a pack per minute; negative while charging)."""
        if len(self._history) < 2:
            return None
        (t0, s0), (t1, s1) = self._history[0], self._history[-1]
        if t1 <= t0:
            return None
        return (s0 - s1) * 60_000 // (t1 - t0)

    # ── ops ───────────────────────────────────────────────────────────────

    def ops(self) -> dict[str, Any]:
        return {"x_battery": self.battery_op}

    def battery_op(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        with body.world.lock:
            return self.command(request)

    def command(self, request: dict[str, Any]) -> dict[str, Any]:
        do = str(request.get("do", "status"))

        def whole(key: str, low: int, high: int) -> int:
            value = request.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"x_battery {key} must be a whole number in [{low}, {high}]")
            return value

        if do == "swap":
            self.pack.swap(whole("soc_ppm", 0, PPM) if "soc_ppm" in request else PPM)
        elif do == "set":
            self.pack.swap(whole("soc_ppm", 0, PPM))
        elif do == "power":
            on = request.get("on")
            if not isinstance(on, bool):
                raise ValueError("x_battery power needs on: true|false")
            self.powered = on
        elif do == "load":
            name = str(request.get("name", ""))
            if not name:
                raise ValueError("x_battery load needs a name")
            ma = whole("ma", 0, 5000)
            if ma:
                self.loads[name] = ma
            else:
                self.loads.pop(name, None)
        elif do == "health":
            self.pack.health_ppm = whole("health_ppm", 100_000, PPM)
        elif do != "status":
            raise ValueError(f"x_battery do {do!r}: status, swap, set, power, load or health")
        self._history.clear()
        return self.snapshot(None)

    def snapshot(self, world: Any) -> dict[str, Any]:
        pack = self.pack
        volts = pack.volts_mv(self.load_ma, self.pg)
        return {
            "model": "world",
            "fed_to_robotd": False,  # E1: robotd still reads the nominal 7.4 V
            "soc_ppm": pack.soc_ppm,
            "percent": pack.soc_ppm * 100 // PPM,
            "volts_mv": volts,
            "ocv_mv": ocv_mv(pack.soc_ppm),
            "current_ma": self.current,
            "load_ma": self.load_ma,
            "charge_ma": pack.charging_ma(self.pg),
            "charging": self.pg and pack.soc_ppm < PPM,
            "pg": self.pg,
            "powered": self.powered,
            "robotd_percent_if_fed": robotd_percent(volts),
            "motor_scale_ppm": motor_scale_ppm(volts, weakness=self.weakness),
            "motor_scale_if_weak_ppm": would_be_scale_ppm(volts),
            "weakness": self.weakness,
            "health_ppm": pack.health_ppm,
            "capacity_mah": pack.capacity_mah,
            "empty": pack.empty,
            "drain_ppm_per_min": self.drain_per_min(),
            "loads_ma": dict(self.loads),
        }


def register(registry: Any) -> WorldBattery:
    """The battery alone as a simhome extension (no nest wired: it never charges)."""
    part = WorldBattery()
    register_part(registry, part, section="battery")
    return part
