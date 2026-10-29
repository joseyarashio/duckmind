"""Live physics behind the simulator page: the official robotd and MuJoCo body, in real time.

Duck Core decides; this plays its intents the way duck-body will on the robot — head
offsets from HOME at 50 Hz, ``robot.mouth``, ``robot.do sit_toggle`` for posture and
``robot.move`` for walks — into ``robotd --sim``, and samples the official body server
(BAM servo models fitted to the real XL330s) for the page. The 3D view then shows what
the real daemon, the real v5 networks and the physics make of each expression: the legs
balancing, the trunk swaying, a stand-up that can fail, a fall that really happens.

The body server runs in the current interpreter, so start the simulator with mujoco and
numpy available (``uv run --with mujoco==3.10.0 --with numpy ...``). Paths come from the
command line or ``DUCKMIND_MICRODUCK``, ``DUCKMIND_MICRODUCK_RL``, ``DUCKMIND_POLICIES``,
``DUCKMIND_ROBOTD`` and ``DUCKMIND_ORT_LIB``. Only child processes started here are
stopped here; ``/run/robotd.sock`` and real robots are never touched.

The official body server is started through :mod:`duckmind.bodyhost` (owner decisions
2026-09-26): the full-body collision model ``robot_allcollisions.xml``, physics stepped in
lockstep with robotd (one 20 ms control period per robotd write), and a stall monitor that
every frame carries as ``"monitor"`` — ``trust`` turns ``"suspect"`` with plain reasons
(``why_zh``) when the physics cannot be trusted. ``DUCKMIND_LIVE_COLLISION``
(``allcollisions`` | ``groundcontact``) and ``DUCKMIND_LIVE_LOCKSTEP`` (``1`` | ``0``) change
those defaults for comparisons without new simulator flags.

Phase 2 (「真的家與真的感測器」, owner decision A1): the body runs in the world server
``duckmind.world.simhome`` in the page's 6×6 home made physical (``default_home``; a duck
without a nest, such as 蛋黃, gets its ``no_nest`` variant: mat and desk, A2) — the same guards,
lockstep and monitor as phase 0's body host, plus the home's walls and furniture in the
physics, the spawn where the layout says and the world ops (``x_place``, ``x_world``). The
household (:mod:`duckmind.world.worldkit`: nest, visible hand, family bodies, world battery,
things) and the sensors (:mod:`duckmind.world.sensors`: the camera behind ``media.frame``, the
ToF behind the official ``tofd``) are loaded into it. ``LiveBody(home=...)`` or
``DUCKMIND_LIVE_HOME`` choose another home (``flat``, a layout name or file); ``bodyhost``
runs phase 0's body host on the flat floor instead (no world ops). ``DUCKMIND_SIMHOME_EXT``
replaces the extension list.

The world's truth (``x_world``) is read a dozen times a second on its own connection
(:meth:`LiveBody.world`); while the world's hand holds the duck up in the air (``hand.held``)
the body is asked to sit with its legs tucked and walks are dropped (owner decision D1), and
Duck Core's posture comes back when it is set down. :meth:`LiveBody.world_call` sends any world
op (``x_place``, ``x_hand``, ``x_people``, …) on a connection of its own.
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckmind
from duckmind import bodyhost
from duckmind.body import BodyIntent
from duckmind.duckcore import content
from duckmind.i18n import msg
from duckmind.robotd import RobotdBodyPort, RobotdOperatorClient
from duckmind.sim_live import (
    SimOracle,
    _free_local_port,
    _policy_config,
    _port_open,
    _socket_open,
    _stop_child,
    _wait_ready,
)
from duckmind.skilltrain.gt_wiring import body_env as skill_table_body_env
from duckmind.skilltrain.gt_wiring import none_slots as skill_table_none_slots
from duckmind.skilltrain.gt_wiring import verified_section as verified_skill_section
from duckmind.skilltrain.table import skill_names_from_env
from duckmind.tmpdir import short_tmp

# robotd's wire order: left leg, neck/head, mouth (index 9), right leg.
WIRE_JOINTS = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
HEAD = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
HOLDS = frozenset({"nap"})  # asleep: the head stays down between stirs
CONTROL_HZ = 50
SAMPLE_HZ = 30
# The page draws every body sample it does not have yet (``LiveBody.frames_since``, fid:page
# 2026-09-28): a 0.3 s head snap must reach the 3D view at its true speed, never eased away. The
# ring keeps the last few seconds of compact samples; a page gets at most FRAMES_PAGE_S of them.
FRAMES_KEEP_S = 3.0
FRAMES_PAGE_S = 1.5
FRAME_JOINTS = tuple(name for name in WIRE_JOINTS if name != "mouth")
REALTIME_WINDOW_S = 2.0  # the realtime reading: simulated per wall second over this long
POSTURE_RETRY_S = 3.0  # re-send a posture change the body has not made by then
RISE_WAIT_S = 6.0  # a walk waits this long for the body to finish standing up
# Owner decisions 2026-09-26 (phase 0): the whole body collides, and the physics advances
# only when robotd has written that tick's targets (see duckmind.bodyhost).
DEFAULT_COLLISION = bodyhost.DEFAULT_COLLISION
DEFAULT_LOCKSTEP = bodyhost.DEFAULT_LOCKSTEP
MONITOR_HZ = 5  # how often the body host's stall monitor is read
MONITOR_STALE_S = 2.0  # a monitor reading older than this is itself a reason for doubt
# Phase 2 (owner decision A1): the page's 6×6 home, made physical, is where the body lives.
DEFAULT_HOME = "default_home"
# DUCKMIND_LIVE_HOME values that run phase 0's body host on the flat floor (no world ops).
BODYHOST_HOMES = frozenset({"bodyhost", "phase0", "none", "off"})
# The world server's extensions: the household (nest, hand, family, battery, things) and the
# sensors (camera via media.frame, ToF via tofd). DUCKMIND_SIMHOME_EXT replaces the list.
WORLD_EXTENSIONS = "duckmind.world.worldkit,duckmind.world.sensors"
WORLD_HZ = 12  # how often the world's truth (x_world) is read for the page
WORLD_TIMEOUT_S = 3.0
# A stall of the host (swap, sleep, a busy machine) is only silence on a body link: robotd's state
# stream, its command line, the oracle's reads. A link that does not answer is tried again with
# these waits (the last one repeats); only a link silent for LINK_FAIL_S is reported as failed, and
# the report is taken back when it speaks again. A link that is over for good (the body or robotd
# exited, the stream closed) is reported at once, as always.
LINK_RETRY_S = (0.2, 0.5, 1.0)
LINK_FAIL_S = 30.0
# robotd's head smoothing for THIS simulator's robotd (fid:motor 2026-09-28): the owner approved
# ``[control] head_alpha`` 0.2 → 1.0 (「加快」; A100 2026-09-28: standing t90 0.46 → 0.34 s, seated
# 0.24 → 0.16 s, 0/10,214 falls; large fast seated moves are rate-limited by the neural motor).
# Set by ``simweb --robotd-head-alpha`` (prototype mode: 1.0); unset = robotd's own default. It
# is written into this body's own robotd.toml (/private/tmp/dm-live-*/): nothing upstream changes.
ROBOTD_HEAD_ALPHA_ENV = "DUCKMIND_ROBOTD_HEAD_ALPHA"
ROBOTD_DEFAULT_HEAD_ALPHA = 0.2  # robotd-params Control::default()


def robotd_head_alpha() -> float | None:
    """The head smoothing asked for this simulator's robotd (``$DUCKMIND_ROBOTD_HEAD_ALPHA``),
    None when not set (robotd keeps its own 0.2). ValueError outside (0, 1]."""
    raw = os.environ.get(ROBOTD_HEAD_ALPHA_ENV, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{ROBOTD_HEAD_ALPHA_ENV} must be in (0, 1], not {raw}")
    return value


def robotd_control_section(head_alpha: float | None) -> str:
    """robotd.toml's ``[control]`` override for this body (empty: robotd's defaults)."""
    if head_alpha is None:
        return ""
    return f"\n[control]\nhead_alpha = {float(head_alpha)!r}\n"


class LinkWatch:
    """How long one body link (robotd's state stream, its command line, the oracle) has been silent.

    A timeout is not a verdict: the host stalls (sleep, swap, a busy machine) and answers again.
    :meth:`missed` says how long to wait before the next try (:data:`LINK_RETRY_S`) and, once the
    link has been silent :data:`LINK_FAIL_S` since it last spoke, that the failure is to be reported;
    :meth:`heard` ends the silence and says whether a report has to be taken back."""

    def __init__(
        self,
        source: str,
        label: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        retry_s: Sequence[float] | None = None,
        fail_s: float | None = None,
    ) -> None:
        self.source = source  # the key of its failure (LiveBody._failures)
        self.label = label  # the words in front of the error text ("body oracle")
        self.clock = clock
        self.retry_s = tuple(LINK_RETRY_S if retry_s is None else retry_s)
        self.fail_s = LINK_FAIL_S if fail_s is None else fail_s
        self.heard_at = clock()
        self.misses = 0
        self.reported = False

    def missed(self) -> tuple[float, bool]:
        """The link did not answer: (seconds to wait before the next try, report the failure now)."""
        self.misses += 1
        wait = self.retry_s[min(self.misses, len(self.retry_s)) - 1]
        report = not self.reported and self.clock() - self.heard_at >= self.fail_s
        self.reported = self.reported or report
        return wait, report

    def heard(self) -> tuple[int, float, bool] | None:
        """The link spoke. None when it had not missed; else (tries it took, seconds it was silent,
        whether a failure had been reported: take it back)."""
        now = self.clock()
        out = (self.misses, now - self.heard_at, self.reported) if self.misses else None
        self.heard_at, self.misses, self.reported = now, 0, False
        return out


DRIVE_S = 0.6  # a direct twist (a demonstration's) lasts this long unless renewed
MONITOR_FIELDS = (
    "collision",
    "lockstep",
    "max_stall_ms_10s",
    "max_stall_ms_60s",
    "robotd_hz",
    "realtime",
    "trust",
    "why_zh",
)


def live_choice(collision: str | None = None, lockstep: bool | None = None) -> tuple[str, bool]:
    """The collision model and stepping to run: given, else ``DUCKMIND_LIVE_COLLISION`` /
    ``DUCKMIND_LIVE_LOCKSTEP``, else the owner's defaults."""
    if collision is None:
        collision = os.environ.get("DUCKMIND_LIVE_COLLISION") or DEFAULT_COLLISION
    if collision not in bodyhost.COLLISIONS:
        raise ValueError(
            f"unknown collision model {collision!r}; known: {sorted(bodyhost.COLLISIONS)}"
        )
    if lockstep is None:
        raw = os.environ.get("DUCKMIND_LIVE_LOCKSTEP", "").strip().lower()
        if raw in {"", "default"}:
            lockstep = DEFAULT_LOCKSTEP
        elif raw in {"1", "on", "true", "yes"}:
            lockstep = True
        elif raw in {"0", "off", "false", "no"}:
            lockstep = False
        else:
            raise ValueError(f"DUCKMIND_LIVE_LOCKSTEP must be 1 or 0, not {raw!r}")
    return collision, lockstep


def live_home(home: str | None = None, *, nest: bool | None = None) -> str | None:
    """The home the world server runs: given, else ``DUCKMIND_LIVE_HOME``, else
    :data:`DEFAULT_HOME` (phase 2, owner decision A1); None (``bodyhost``, ``phase0``, ``none``,
    ``off``) for phase 0's body host on the flat floor. ``nest=False`` (a duck without a nest,
    such as 蛋黃): the home's ``no_nest`` variant when it has one and no variant was named."""
    if home is None:
        home = os.environ.get("DUCKMIND_LIVE_HOME") or ""
    home = home.strip() or DEFAULT_HOME
    if home.lower() in BODYHOST_HOMES:
        return None
    if nest is False:
        home = without_nest(home)
    return home


def without_nest(home: str) -> str:
    """``home`` for a duck without a nest: ``name,no_nest`` when the layout offers that variant
    and no variant was named; anything else (a variant given, a file, ``flat``) as it is."""
    if "," in home or home.endswith(".json") or home == "flat":
        return home
    return f"{home},no_nest" if _offers_no_nest(home) else home


@functools.lru_cache(maxsize=32)
def _offers_no_nest(home: str) -> bool:
    try:
        from duckmind.world import homelayout

        return "no_nest" in homelayout.variants(home)
    except Exception:  # not a named layout: leave it as it is
        return False


POSE_JOINTS_ENV = (
    "DUCKMIND_ROBOTD_POSE_JOINTS"  # 'off' pretends the robotd has none (the sit-only play dead)
)
_POSE_JOINTS_MARK = b"robot.poseJoints"


@functools.lru_cache(maxsize=8)
def _has_pose_joints(path: str, mtime_ns: int) -> bool:
    keep = len(_POSE_JOINTS_MARK) - 1
    tail = b""
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            if _POSE_JOINTS_MARK in tail + chunk:
                return True
            tail = chunk[-keep:]
    return False


def robotd_has_pose_joints(robotd: str | os.PathLike[str]) -> bool:
    """Whether this robotd build answers ``robot.poseJoints`` (the simulator-only build of
    docs/design/neural-duck/PLAY-DEAD-2026-10-11.md: the method's name is in the binary). The official 0.15.0
    robotd has none: play dead then stays the sitting one. ``DUCKMIND_ROBOTD_POSE_JOINTS=off`` forces that."""
    if os.environ.get(POSE_JOINTS_ENV, "").strip().lower() in ("off", "0", "no", "false"):
        return False
    try:
        path = Path(robotd)
        return _has_pose_joints(str(path), path.stat().st_mtime_ns)
    except OSError:
        return False


def find_tofd(paths: LivePaths) -> Path | None:
    """The official tofd built next to robotd (owner decision F3), or in the checkout's
    target folder; None when it was never built (the sensors then run no tofd)."""
    candidates = [
        paths.robotd.parent / "tofd",
        paths.microduck / "target" / "aarch64-apple-darwin" / "debug" / "tofd",
        paths.microduck / "target" / "debug" / "tofd",
    ]
    return next((path for path in candidates if path.is_file()), None)


def unknown_monitor(collision: str, lockstep: bool, why: str) -> dict[str, Any]:
    """A monitor with no numbers yet (or none to be had): suspect, and why."""
    return {
        "collision": collision,
        "lockstep": lockstep,
        "max_stall_ms_10s": None,
        "max_stall_ms_60s": None,
        "robotd_hz": None,
        "realtime": None,
        "trust": "suspect",
        "why_zh": [why],
    }


def _log_tail(path: Path, lines: int = 6) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return "(no log)"
    return " / ".join(line.strip() for line in text[-lines:]) or "(empty log)"


@dataclass(frozen=True)
class LivePaths:
    microduck: Path
    microduck_rl: Path
    policies: Path
    robotd: Path
    ort_lib: Path

    @classmethod
    def from_env(cls) -> LivePaths | None:
        names = (
            "DUCKMIND_MICRODUCK",
            "DUCKMIND_MICRODUCK_RL",
            "DUCKMIND_POLICIES",
            "DUCKMIND_ROBOTD",
            "DUCKMIND_ORT_LIB",
        )
        values = [os.environ.get(name) for name in names]
        if not all(values):
            return None
        return cls(*(Path(str(value)).expanduser() for value in values))

    def missing(self) -> list[str]:
        checks = {
            "microduck": self.microduck.is_dir(),
            "microduck_rl body server": (
                self.microduck_rl / "src/mjlab_microduck/sim/body_server.py"
            ).is_file(),
            "policies": (self.policies / "velstand.onnx").is_file(),
            "robotd": self.robotd.is_file(),
            "ONNX Runtime": self.ort_lib.is_file(),
        }
        return [name for name, ok in checks.items() if not ok]


class ClipPlayer:
    """Duck Core expression intents → per-tick head offsets and mouth opening (duck-body)."""

    def __init__(self, side: int = 1) -> None:
        self.side = side
        self.current: dict[str, Any] | None = None
        self.rest: dict[str, Any] | None = None  # the held nap pose while it sleeps
        self.queue: deque[dict[str, Any]] = deque(maxlen=2)

    def push(self, motion: dict[str, Any], now: float) -> None:
        clip = str(motion.get("clip"))
        if clip not in content.CLIPS:
            return
        entry = {
            "clip": clip,
            "amplitude": float(motion.get("amplitude", 10_000)) / 10_000,
            "tempo": max(0.5, float(motion.get("tempo", 10_000)) / 10_000),
            "weak": bool(motion.get("weak")),
        }
        # An idle glance gives way to the next glance and to anything the duck really does.
        idle = (
            self.current is None or self.current["clip"] in HOLDS or bool(self.current.get("weak"))
        )
        if motion.get("weak") and (not idle or self.queue or self.rest is not None):
            return  # an idle glance never delays (or wakes) anything the duck really does
        if clip in HOLDS:
            self.rest = entry  # asleep: every stir settles back into this pose
        if idle:
            self._start(entry, now)
        else:
            self.queue.append(entry)

    def clear(self) -> None:
        self.current = None
        self.rest = None
        self.queue.clear()

    def wake(self) -> None:
        self.rest = None
        if self.current and self.current["clip"] in HOLDS:
            self.current = None

    def _start(self, entry: dict[str, Any], now: float) -> None:
        self.current = dict(entry, start=now)

    def targets(self, now: float) -> tuple[dict[str, float], float]:
        """Offsets from HOME (radians) for the four head joints, and mouth opening."""
        frame: dict[str, float] = {}
        if self.current is not None:
            frames = content.CLIPS[self.current["clip"]]
            elapsed_ms = (now - self.current["start"]) * 1000.0 * self.current["tempo"]
            if elapsed_ms >= frames[-1][0] and self.current["clip"] not in HOLDS:
                self.current = None
                if self.queue:
                    self._start(self.queue.popleft(), now)
                    return self.targets(now)
                if self.rest is not None:
                    # A stir is over: back into the held sleep pose (at its last frame).
                    held = content.CLIPS[self.rest["clip"]][-1][0] / 1000.0 / self.rest["tempo"]
                    self.current = dict(self.rest, start=now - held)
                    return self.targets(now)
            else:
                raw = content.frame_at(frames, min(elapsed_ms, frames[-1][0]))
                amplitude = self.current["amplitude"]
                frame = {
                    key: value * amplitude if key != "m" else value for key, value in raw.items()
                }
        goal = content.head_targets(frame, self.side)
        offsets = {name: goal[name] - content.HEAD_HOME[name] for name in HEAD}
        return offsets, goal["mouth"]


class LiveBody:
    def __init__(
        self,
        paths: LivePaths,
        *,
        side: int = 1,
        seated: bool = True,
        send_head: bool = True,
        send_idle_move: bool = False,
        collision: str | None = None,
        lockstep: bool | None = None,
        home: str | None = None,
        nest: bool | None = None,
        tof: bool | None = None,
    ) -> None:
        self.paths = paths
        self.seated_start = seated
        # None: duckmind.bodyhost (phase 0); a home: duckmind.world.simhome (live_home). A
        # duck without a nest gets the home's no_nest variant; ``tof``: whether its head has
        # the ToF module (DUCKMIND_SIM_TOF for the sensors; None: their default).
        self.home = live_home(home, nest=nest)
        self.tof = tof
        # Which robot model the physics collides with, and whether it steps in lockstep with
        # robotd (None: the environment's choice, else the owner's defaults; live_choice).
        self.collision, self.lockstep = live_choice(collision, lockstep)
        self.send_head = send_head
        # robotd zeroes a stale twist after 500 ms by itself, so a zero twist is only sent
        # while a walk is ending; streaming zeros is optional (diagnostics).
        self.send_idle_move = send_idle_move
        self._walking = False
        self.player = ClipPlayer(side)
        self._lock = threading.Lock()  # guards the command connection and shared state
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._state_dir: Path | None = None
        self._body: subprocess.Popen[bytes] | None = None
        self._robotd: subprocess.Popen[bytes] | None = None
        self._port: RobotdBodyPort | None = None
        self._operator: RobotdOperatorClient | None = None
        self._oracle: SimOracle | None = None
        # does this robotd answer robot.poseJoints (the simulator-only build)? play dead lies down when it does
        self.pose_joints_ok = robotd_has_pose_joints(paths.robotd)
        # The posture Duck Core wants, kept until the body gets there: a toggle sent while
        # the policies are busy (rising, mid-step) can be ignored, so it is re-sent.
        self._posture_request: str | None = None
        self._last_toggle = -math.inf
        # vx, vyaw, seconds, until (monotonic; None until the body is up), give-up time
        self._walk: tuple[float, float, float, float | None, float] | None = None
        self._awake = True
        self.policy = "starting"
        self.fallen = False
        self.error: str | None = None
        # The failures reported by links that may still come back (LinkWatch.source -> the error
        # text); ``error`` shows the first and is cleared again when that link speaks. An error
        # set any other way (the body never started, a thread that ended for good) is never cleared.
        self._failures: dict[str, str] = {}
        self._error_lock = threading.Lock()
        self.latest: dict[str, Any] | None = None
        # Simulated seconds per wall-clock second over the last few samples. A policy
        # stepped slower than real time misbehaves, so the page must know.
        self.realtime_factor = 1.0
        self._clock: deque[tuple[float, float]] = deque(maxlen=400)
        # How often the body server is read (SAMPLE_HZ; the bridge sets 50 in prototype mode:
        # every control tick), the ring of compact samples the page plays (frames_since), and
        # the beak the neural duck commands (its driver sets it: the physics model has no mouth).
        self.sample_hz = float(SAMPLE_HZ)
        self._frames: deque[dict[str, Any]] = deque(maxlen=int(FRAMES_KEEP_S * 60) + 20)
        self._frames_lock = threading.Lock()
        self.mouth_cmd: float | None = None
        self._calls: list[dict[str, Any]] = []  # calls voiced since the last sample (voice())
        # The body host's stall monitor (bodyhost x_monitor), read by _watch; its numbers
        # and verdict go into status() and every frame.
        self.body_port: int | None = None
        self.host_info: dict[str, Any] | None = None
        self._host: bodyhost.HostClient | None = None
        self.monitor = unknown_monitor(
            self.collision, self.lockstep, msg("livebody.LiveBody.__init__.1771e213")
        )
        self._monitor_at = -math.inf
        # The world's truth (x_world, phase 2), read by _world_watch; None before the first
        # answer or without a world (the body host). Its own connection for world ops.
        self.world_state: dict[str, Any] | None = None
        self.world_at = -math.inf
        self._world_lock = threading.Lock()
        self._world_client: bodyhost.HostClient | None = None
        # The world's hand holds the duck up: it sits with its legs tucked, walks wait (D1).
        self._held = False
        # The posture Duck Core last asked for (restored when the hand sets the duck down).
        self._wanted: str | None = None
        # A demonstration's direct twist (vx, vy, vyaw, until) and a skill to send once.
        self._drive: tuple[float, float, float, float] | None = None
        self._skill: str | None = None
        self.skill_ack: dict[str, Any] | None = None
        # Passive: another robotd writer has the body (duck-body during 「用眼睛做」, phase 3;
        # spec §6: one writer at a time). The control loop then sends robotd nothing — no
        # head, mouth, move, posture or skill — and drops any walk; state is still read.
        self.passive = False
        # robotd's [control] head_alpha for this body (start() writes it; fid:motor)
        self.robotd_head_alpha = ROBOTD_DEFAULT_HEAD_ALPHA
        # the one-shot skills this body's robotd answers robot.do for (its subscribe ack)
        self.robotd_skills: tuple[str, ...] = ()

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self, timeout_s: float = 60.0) -> None:
        missing = self.paths.missing()
        if missing:
            raise FileNotFoundError("live body needs: " + ", ".join(missing))
        port = _free_local_port()
        state = Path(tempfile.mkdtemp(prefix="dm-live-", dir=short_tmp()))
        self._state_dir = state
        params = state / "robotd.toml"
        head_alpha = robotd_head_alpha()
        # what this body's robotd smooths the head with (the neural motor reads it: fid:motor)
        self.robotd_head_alpha = ROBOTD_DEFAULT_HEAD_ALPHA if head_alpha is None else head_alpha
        # Stage-1 trained one-shot skills (tr:harness wiring, 2026-09-28): the [[policy.skill]]
        # table of $DUCKMIND_ROBOTD_SKILLS (a skills.json) for THIS simulator's robotd only;
        # nothing when unset. No upstream file changes. (s1b:fix) Only when every network the
        # table runs — the skills AND the walk network its abort and hand-backs use — is the
        # one it was gated with (sha256 pins): an OTA-changed velstand refuses to start.
        params.write_text(
            _policy_config(self.paths.policies, none_slots=skill_table_none_slots())
            + robotd_control_section(head_alpha)
            + verified_skill_section()
        )
        sock = state / "robotd.sock"
        body_log = (state / "body.log").open("w")
        robot_log = (state / "robotd.log").open("w")
        # The official body server, hosted by duckmind.bodyhost: the chosen collision model,
        # lockstep stepping and the stall monitor (x_ ops on the same port; robotd never
        # sends them). duckmind itself must be importable from microduck_rl's cwd too.
        self._body = subprocess.Popen(
            self._body_command(port),
            cwd=self.paths.microduck_rl,
            env=self._body_env(state),
            stdout=body_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(self._body, lambda: _port_open(port), deadline_s=30, label="duck-body")
        except (RuntimeError, TimeoutError) as error:
            # bodyhost says why it refused (an upstream change, a scene that differs) in its log.
            raise RuntimeError(f"{error}: {_log_tail(state / 'body.log')}") from error
        self.body_port = port
        self._host = bodyhost.HostClient(port)
        self.host_info = self._host.info()
        self._robotd = subprocess.Popen(
            [
                str(self.paths.robotd),
                "--sim",
                f"127.0.0.1:{port}",
                "--params",
                str(params),
                "--socket",
                str(sock),
            ],
            cwd=self.paths.microduck,
            env={
                **os.environ,
                "ORT_DYLIB_PATH": str(self.paths.ort_lib),
                "DUCK_RUNTIME_DIR": str(state),
                "RUST_LOG": "warn",
            },
            stdout=robot_log,
            stderr=subprocess.STDOUT,
        )
        _wait_ready(self._robotd, lambda: _socket_open(sock), deadline_s=30, label="robotd")
        self._port = RobotdBodyPort(sock, backend="simulator", hz=25)
        if self._port.capabilities.get("walk") != "velstand.onnx":
            raise RuntimeError(f"unexpected walking policy: {self._port.capabilities}")
        self.robotd_skills = tuple(str(s) for s in self._port.capabilities.get("skills") or ())
        missing_skills = sorted(set(skill_names_from_env()) - set(self.robotd_skills))
        if missing_skills:
            raise RuntimeError(f"robotd did not load the trained skills {missing_skills}")
        self._operator = RobotdOperatorClient(sock)
        self._oracle = SimOracle(port)
        threads = [self._read_state, self._control, self._sample, self._watch]
        if self.home:
            threads.append(self._world_watch)
        for target in threads:
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        ack = self._operator.enable(True)
        if not ack.accepted:
            raise RuntimeError(f"robotd refused enable: {ack.reason}")
        # A seated body rises on enable (robotd's boot detector); wait for the balance
        # network, then sit back down so the page starts on a seated desk pet.
        self._wait(lambda: self.policy in {"walk", "stand"} and not self.fallen, timeout_s, "stand")
        if self.seated_start:
            self.request_posture("sit")
            self._wait(lambda: self.policy == "sit", 15.0, "sit")

    def _body_command(self, port: int) -> list[str]:
        """How the official body server is started: through duckmind.bodyhost, or through
        the world server duckmind.world.simhome when a home is asked for."""
        command = [
            sys.executable,
            "-m",
            "duckmind.world.simhome" if self.home else "duckmind.bodyhost",
            "--headless",
            "--keyframe",
            "SIT" if self.seated_start else "HOME",
            "--port",
            str(port),
            "--collision",
            self.collision,
            "--lockstep" if self.lockstep else "--no-lockstep",
        ]
        if self.home:
            command += ["--home", self.home]
        return command

    def _body_env(self, state: Path) -> dict[str, str]:
        """The body server's environment: this repo importable from microduck_rl's cwd and, in
        a home, the household and sensors extensions with their settings (anything already
        set in this process's environment wins)."""
        package_root = str(Path(duckmind.__file__).resolve().parent.parent)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(self.paths.microduck_rl / "src"), package_root)),
        }
        # tr:gate: the servo command delay the installed skill table was gated with (its
        # "body" block; an explicit DUCKMIND_BODY_COMMAND_DELAY_MS wins; no table: unchanged)
        env = skill_table_body_env(env)
        if not self.home:
            return env
        env.setdefault("DUCKMIND_SIMHOME_EXT", WORLD_EXTENSIONS)
        # The camera's and the ToF's sockets live with this body (removed with it).
        env.setdefault("DUCKMIND_SENSORS_DIR", str(state / "sensors"))
        tofd = find_tofd(self.paths)
        if tofd is not None:
            env.setdefault("DUCKMIND_TOFD", str(tofd))
        if self.tof is not None:
            # F2: a duck without the module still runs tofd, which says it is not there.
            env.setdefault("DUCKMIND_SIM_TOF", "1" if self.tof else "0")
        return env

    def _wait(self, check: Any, timeout_s: float, label: str) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError(self.error)
            if check():
                return
            time.sleep(0.05)
        raise TimeoutError(f"live body never reached {label} (policy {self.policy})")

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)
        try:
            if self._operator is not None:
                self._operator.enable(False)
        except Exception:
            pass
        for closer in (self._oracle, self._operator, self._port, self._host, self._world_client):
            try:
                if closer is not None:
                    closer.close()
            except Exception:
                pass
        _stop_child(self._robotd)
        _stop_child(self._body)
        if self._state_dir is not None and not os.environ.get("DUCKMIND_KEEP_LIVE_LOGS"):
            shutil.rmtree(self._state_dir, ignore_errors=True)

    # ── intents from Duck Core ──────────────────────────────────────────

    def set_side(self, side: int) -> None:
        self.player.side = 1 if side >= 0 else -1

    @property
    def walking(self) -> bool:
        """A walk (or turn) is asked for or under way: the body is about to move or moves."""
        return self._walk is not None or self._walking

    def request_posture(self, target: str) -> None:
        with self._lock:
            self._posture_request = target

    @property
    def held(self) -> bool:
        """The world's hand holds the duck up in the air (x_world ``hand.held``)."""
        return self._held

    def drive(self, vx: float, vy: float, vyaw: float, seconds: float = DRIVE_S) -> None:
        """A direct twist for ``seconds`` (a demonstration's steering, never Duck Core's)."""
        with self._lock:
            if self._held:
                return
            self._drive = (float(vx), float(vy), float(vyaw), time.monotonic() + seconds)

    def pose_joints(
        self, targets: Sequence[float | None], off: Sequence[str], *, gain: int, ramp_s: float
    ) -> bool:
        """The operator's ``robot.poseJoints`` (play dead's scripted pose); False when this robotd has none or refuses."""
        operator = self._operator
        if operator is None or not self.pose_joints_ok:
            return False
        try:
            return bool(operator.pose_joints(targets, off, gain=gain, ramp_s=ramp_s).accepted)
        except Exception as error:  # an older robotd answers 'unknown method'
            self.error = f"poseJoints: {error}"
            self.pose_joints_ok = False
            return False

    def operator_init(self) -> bool:
        """The operator's ``robot.init`` (torque back on everywhere, the ramp home)."""
        operator = self._operator
        if operator is None:
            return False
        try:
            return bool(operator.init().accepted)
        except Exception as error:
            self.error = f"init: {error}"
            return False

    def operator_enable(self, on: bool = True) -> bool:
        """The operator's ``robot.enable`` (the walking policy back on)."""
        operator = self._operator
        if operator is None:
            return False
        try:
            return bool(operator.enable(bool(on)).accepted)
        except Exception as error:
            self.error = f"enable: {error}"
            return False

    def skill(self, name: str) -> None:
        """One official one-shot skill (``robot.do``: ground_pick, kick_left, kick_right)."""
        with self._lock:
            self._skill = str(name)

    def consume(self, motions: list[dict[str, Any]]) -> None:
        """Feed ``api.feed`` motion dicts: clips, posture, walks and power."""
        now = time.monotonic()
        for motion in motions:
            if motion.get("clip"):
                with self._lock:
                    if self._awake:
                        self.player.push(motion, now)
            elif motion.get("posture") in {"stand", "sit"} or motion.get("sync") in {
                "stand",
                "sit",
            }:
                # Switched off, it sits and stays down: no posture requests, no walks. Held
                # up by the hand it sits; Duck Core's posture comes back when it is set down.
                wanted = str(motion.get("posture") or motion.get("sync"))
                if self._awake:
                    self._wanted = wanted
                    if not self._held:
                        self.request_posture(wanted)
            elif isinstance(motion.get("walk"), dict):
                if not self._awake or self._held:
                    continue
                walk = motion["walk"]
                vx = float(walk.get("vx", 0.0))
                vyaw = float(walk.get("vyaw", 0.0))
                seconds = float(walk.get("seconds", 0.0))
                bounds = content.LOCOMOTION
                top_vx = float(cast(tuple[float, float], bounds["walk_vx"])[1])
                top_yaw = float(cast(float, bounds["turn_vyaw"]))
                bounded_vx = min(max(vx, 0.0), top_vx)
                bounded_yaw = max(-top_yaw, min(top_yaw, vyaw))
                seconds = min(seconds, float(cast(float, bounds["walk_s_max"])))
                if self.policy == "sit":
                    self.request_posture("stand")  # its stand-up was skipped (time jumped)
                with self._lock:
                    # The walk's clock starts once the body is standing: rising takes the
                    # policies a second or two, which would otherwise eat the walk.
                    self._walk = (bounded_vx, bounded_yaw, seconds, None, now + RISE_WAIT_S)
            elif motion.get("power") == "off":
                with self._lock:
                    self._awake = False
                    self.player.clear()
                    self._walk = None
                    self._drive = None
                    self._posture_request = "sit"
                    self._wanted = "sit"
            elif motion.get("power") == "on":
                with self._lock:
                    self._awake = True
                    self.player.wake()
            elif motion.get("wake"):
                with self._lock:
                    self.player.wake()

    # ── threads ─────────────────────────────────────────────────────────

    def _link_failed(self, link: LinkWatch, error: BaseException) -> None:
        """A link has been silent for LINK_FAIL_S: the body is reported failed (``error``)."""
        text = f"{link.label}: {error}"
        with self._error_lock:
            self._failures[link.source] = text
            if self.error is None:
                self.error = text
        print(
            f"live body: {link.label} silent {link.clock() - link.heard_at:.0f} s: {error}",
            flush=True,
        )

    def _link_heard(self, link: LinkWatch) -> None:
        """A link spoke after a miss: take its report back, say how long it was silent."""
        back = link.heard()
        if back is None:
            return
        tries, silent_s, reported = back
        if reported:
            with self._error_lock:
                text = self._failures.pop(link.source, None)
                if text is not None and self.error == text:
                    self.error = next(iter(self._failures.values()), None)
        print(
            f"live body: {link.label} answers again after {silent_s:.1f} s ({tries} tries)"
            + (", failure cleared" if reported else ""),
            flush=True,
        )

    def _process_gone(self, process: subprocess.Popen[bytes] | None) -> bool:
        return process is not None and process.poll() is not None

    def _read_state(self) -> None:
        assert self._port is not None
        link = LinkWatch("state", "robotd state")
        while not self._stop.is_set():
            try:
                observed = self._port.observe()
            except TimeoutError as error:  # silence: the host stalls and answers again
                if self._stop.is_set():
                    return
                if self._process_gone(self._robotd):
                    self.error = f"robotd state: {error}"
                    return
                wait, report = link.missed()
                if report:
                    self._link_failed(link, error)
                self._stop.wait(wait)
                continue
            except Exception as error:  # the daemon went away
                if not self._stop.is_set():
                    self.error = f"robotd state: {error}"
                return
            self._link_heard(link)
            self.policy = observed.policy
            self.fallen = observed.safety.fallen

    def _control(self) -> None:
        assert self._port is not None
        period = 1.0 / CONTROL_HZ
        next_tick = time.monotonic()
        link = LinkWatch("command", "robotd command")
        while not self._stop.is_set():
            now = time.monotonic()
            if self.passive:
                with self._lock:
                    self._walk = self._drive = None
                    self._walking = False
                    self._skill = None
                next_tick = time.monotonic() + period
                time.sleep(period)
                continue
            try:
                with self._lock:
                    offsets, mouth = (
                        self.player.targets(now) if self._awake else ({n: 0.0 for n in HEAD}, 0.0)
                    )
                    if self._held:
                        # Up in the air in the world's hand: legs tucked (it sits), no walks.
                        self._walk = self._drive = None
                        self._posture_request = "sit"
                    posture = self._posture_request
                    walk = self._walk
                    if walk is not None and walk[3] is None:
                        if self.policy in {"walk", "stand"} and not self.fallen:
                            walk = self._walk = (*walk[:3], now + walk[2], walk[4])
                        elif now >= walk[4]:
                            self._walk = walk = None  # never stood up: drop it
                    if walk is not None and walk[3] is not None and now >= walk[3]:
                        self._walk = walk = None
                    moving = walk is not None and walk[3] is not None
                    drive = self._drive
                    if drive is not None and now >= drive[3]:
                        self._drive = drive = None
                    if self.send_head:
                        self._port.dispatch_intent(BodyIntent("robot.head", offsets))
                        self._port.dispatch_intent(BodyIntent("robot.mouth", {"open": mouth}))
                    vx, vy, vyaw = (
                        (walk[0], 0.0, walk[1]) if walk is not None and moving else (0.0, 0.0, 0.0)
                    )
                    if drive is not None:  # a demonstration steers (never Duck Core)
                        vx, vy, vyaw = drive[:3]
                        moving = True
                    if moving or self._walking or self.send_idle_move:
                        self._port.dispatch_intent(
                            BodyIntent("robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw})
                        )
                    self._walking = moving
                    skill, self._skill = self._skill, None
                    if skill is not None and not self._held:
                        try:
                            ack = self._port.dispatch_intent(
                                BodyIntent("robot.do", {"skill": skill})
                            )
                            self.skill_ack = {"skill": skill, "accepted": ack.accepted}
                        except Exception as error:  # a refused skill never stops the body
                            self.skill_ack = {
                                "skill": skill,
                                "accepted": False,
                                "error": str(error),
                            }
                    if posture is not None and not moving and not self.fallen:
                        sitting = self.policy == "sit"
                        upright = self.policy in {"walk", "stand"}
                        wrong = (posture == "stand" and sitting) or (posture == "sit" and upright)
                        if not wrong:
                            self._posture_request = None if not walk else posture
                        elif now - self._last_toggle >= POSTURE_RETRY_S:
                            self._last_toggle = now
                            self._port.dispatch_intent(
                                BodyIntent("robot.do", {"skill": "sit_toggle"})
                            )
            except TimeoutError as error:  # silence: robotd stalls with the host and answers again
                if self._stop.is_set():
                    return
                if self._process_gone(self._robotd):
                    self.error = f"robotd command: {error}"
                    return
                wait, report = link.missed()
                if report:
                    self._link_failed(link, error)
                self._stop.wait(wait)
                next_tick = time.monotonic()
                continue
            except Exception as error:
                if not self._stop.is_set():
                    self.error = f"robotd command: {error}"
                return
            self._link_heard(link)
            next_tick += period
            slack = next_tick - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.monotonic()

    def _sample(self) -> None:
        assert self._oracle is not None
        link = LinkWatch("oracle", "body oracle")
        # A fixed rate (``sample_hz``, read each pass: the bridge may raise it), not a sleep
        # after each read: the page's frames come at an even pace, and a late read never
        # starts a burst of catch-up reads.
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                if self._oracle.lost:  # a failed read drops its connection (and any late answer)
                    self._oracle.reconnect()
                snap = self._oracle.snapshot()
            except (OSError, ValueError) as error:  # silence or a broken link: the host may be back
                if self._stop.is_set():
                    return
                if self._process_gone(self._body):
                    self.error = f"body oracle: {error}"
                    return
                wait, report = link.missed()
                if report:
                    self._link_failed(link, error)
                self._stop.wait(wait)
                next_tick = time.monotonic()
                continue
            except Exception as error:  # the host answered with an error: not a stall
                if not self._stop.is_set():
                    self.error = f"body oracle: {error}"
                return
            self._link_heard(link)
            wall = time.time()
            positions = snap["positions"]
            sim_t = float(snap.get("sim_time", 0.0))
            read_at = time.monotonic()
            self._clock.append((read_at, sim_t))
            while len(self._clock) > 10 and read_at - self._clock[0][0] > REALTIME_WINDOW_S:
                self._clock.popleft()
            if len(self._clock) >= 10:
                (wall0, sim0), (wall1, sim1) = self._clock[0], self._clock[-1]
                self.realtime_factor = (sim1 - sim0) / max(1e-6, wall1 - wall0)
            # The fields (t, joints, trunk, quat from the official body server's `read`,
            # body_server.Body.sensors):
            # - t: simulated seconds (MuJoCo data.time).
            # - positions: simulator ground truth in robotd wire order (joint 9 is the mouth only
            #   when the cached world explicitly reports a jaw actuator; otherwise never infer it).
            # - trunk: the world position in metres of the trunk_base body's frame origin (the
            #   free joint's qpos[0:3]): x, y on the floor, z up from the floor plane (z = 0).
            #   In a home (phase 2) these are the home's own world coordinates (the layout's;
            #   default_home: the same as Duck Core's room); on the body host's flat floor the
            #   duck starts at x = y = 0 facing +x. z is ~0.06 seated, ~0.12 standing (plus
            #   whatever it stands on: the nest pad, or a hand holding it up).
            # - quat: that body's orientation in the world, a unit quaternion in MuJoCo order
            #   (w, x, y, z), body to world (qpos[3:7]); the IMU site sits in the trunk with no
            #   rotation, so it is also what the IMU reports as `quat`.
            #   No robot can measure trunk or quat: they are ground truth for the page's
            #   drawing and for evaluators, never for Duck Core.
            # - policy, fallen: robotd's robot.state; realtime: simulated per wall second over
            #   the last ~2 s of these samples.
            # - monitor: bodyhost's stall monitor and its verdict (monitor_view).
            mouth_rad, mouth_source = self._mouth_readback(snap)
            self.latest = {
                "t": float(snap.get("sim_time", 0.0)),
                "mouth_measured_rad": mouth_rad,
                "mouth_source": mouth_source,
                "joints": {
                    name: float(positions[index])
                    for index, name in enumerate(WIRE_JOINTS)
                    if name != "mouth"
                },
                "trunk": [float(value) for value in snap.get("trunk", [0.0, 0.0, snap["trunk_z"]])],
                "quat": [
                    float(value) for value in snap.get("imu", {}).get("quat", [1.0, 0.0, 0.0, 0.0])
                ],
                "policy": self.policy,
                "fallen": self.fallen,
                "realtime": round(self.realtime_factor, 3),
                "monitor": self.monitor_view(),
            }
            self._keep_frame(self.latest, wall)
            next_tick += 1.0 / max(1.0, float(self.sample_hz))
            slack = next_tick - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.monotonic()

    def _mouth_readback(self, snap: dict[str, Any]) -> tuple[float | None, str]:
        """Read the simulated beak only when the cached world explicitly has a measured jaw."""
        world = self.world()  # cached, freshness-checked; never ask the world on this sample path
        neuro = (world or {}).get("neuro")
        capabilities = neuro.get("capabilities", {}) if isinstance(neuro, dict) else {}
        positions = snap.get("positions")
        if (
            capabilities.get("jaw_present") is True
            and isinstance(positions, list)
            and len(positions) == len(WIRE_JOINTS)
        ):
            value = positions[WIRE_JOINTS.index("mouth")]
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                return float(value), "measured_sim"
        return None, "commanded_fallback"

    def _keep_frame(self, latest: dict[str, Any], wall: float) -> None:
        """One compact sample for the page's ring (only a new one: in lockstep a read between
        two physics steps gives the same sample again). ``t`` simulated seconds, ``w`` the wall
        clock of the read (epoch seconds: the page plays every sample at its own moment), ``j``
        the FRAME_JOINTS in that order (rad), ``p`` the trunk (m), ``q`` its quaternion (w, x,
        y, z), ``m`` the beak the neural duck commands (0..1; only when a driver sets it), ``v``
        the calls it voiced since the sample before (:meth:`voice`)."""
        t = round(float(latest["t"]), 4)
        joints = latest["joints"]
        frame: dict[str, Any] = {
            "t": t,
            "w": round(wall, 4),
            "j": [round(float(joints.get(name, 0.0)), 4) for name in FRAME_JOINTS],
            "p": [round(float(value), 5) for value in latest["trunk"]],
            "q": [round(float(value), 5) for value in latest["quat"]],
        }
        mouth_rad = latest.get("mouth_measured_rad")
        if (
            latest.get("mouth_source") == "measured_sim"
            and isinstance(mouth_rad, (int, float))
            and math.isfinite(float(mouth_rad))
        ):
            # robotd travel is −5° closed to +30° open; the 3D beak consumes a 0..1 fraction.
            closed, opened = math.radians(-5.0), math.radians(30.0)
            frame["m"] = round(
                max(0.0, min(1.0, (float(mouth_rad) - closed) / (opened - closed))), 3
            )
            frame["m_source"] = "measured_sim"
        elif self.mouth_cmd is not None:
            frame["m"] = round(max(0.0, min(1.0, float(self.mouth_cmd))), 3)
            frame["m_source"] = "commanded_fallback"
        else:
            frame["m_source"] = "unavailable"
        with self._frames_lock:
            if self._frames:
                last_t = float(self._frames[-1]["t"])
                if t == last_t:
                    return  # (the physics did not step since the last read: the same sample)
                if t < last_t:
                    self._frames.clear()  # (a new body: its clock started over)
            if self._calls:
                frame["v"], self._calls = self._calls, []
            self._frames.append(frame)
            while self._frames and t - float(self._frames[0]["t"]) > FRAMES_KEEP_S:
                self._frames.popleft()

    def voice(
        self,
        tag: str,
        level: float = 0.5,
        parts: Sequence[str] | None = None,
        lib: str | None = None,
    ) -> None:
        """A call the neural duck voiced this tick (its driver): it rides on the next body
        sample (``v``), so the page sounds it with the beak that opened for it. (The robot's
        speaker is not simulated; the page voices it.) ``parts``: the sound tags a learned phrase
        or the song is made of (the page's bubble); ``lib``: the voice library the call comes from
        (docs/design/GROWTH-G5-2026-10-11.md)."""
        with self._frames_lock:
            call: dict[str, Any] = {"tag": str(tag), "level": round(float(level), 3)}
            if parts:
                call["parts"] = [str(part) for part in parts]
            if lib:
                call["lib"] = str(lib)
            self._calls.append(call)
            del self._calls[:-8]

    def frames_since(
        self, since_t: float | None = None, limit_s: float = FRAMES_PAGE_S
    ) -> list[dict[str, Any]]:
        """The ring's samples after simulated time ``since_t`` (the newest the page has), at
        most the last ``limit_s`` seconds of them; all of those without ``since_t`` or when it
        is not in this body's time (a restarted body starts its clock over)."""
        with self._frames_lock:
            frames = list(self._frames)
        if not frames:
            return []
        newest = float(frames[-1]["t"])
        floor = newest - limit_s
        if since_t is not None and floor <= since_t <= newest:
            floor = since_t
        return [frame for frame in frames if float(frame["t"]) > floor]

    def _watch(self) -> None:
        """Read the body host's stall monitor a few times a second (its own connection)."""
        period = 1.0 / MONITOR_HZ
        while not self._stop.is_set():
            try:
                if self._host is None:
                    self._host = bodyhost.HostClient(int(self.body_port or 0))
                reading = self._host.monitor()
                self.monitor = {**reading, "why_zh": list(reading.get("why_zh") or [])}
            except Exception:
                if self._stop.is_set():
                    return
                gone = self._body is not None and self._body.poll() is not None
                why = (
                    msg("livebody.LiveBody._watch.82fa94a5")
                    if gone
                    else msg("livebody.LiveBody._watch.12de7740")
                )
                self.monitor = unknown_monitor(self.collision, self.lockstep, why)
                host, self._host = self._host, None  # reconnect on the next round
                try:
                    if host is not None:
                        host.close()
                except Exception:
                    pass
            self._monitor_at = time.monotonic()
            self._stop.wait(period)

    def _world_watch(self) -> None:
        """Read the world's truth (x_world) about WORLD_HZ times a second on its own
        connection; note when the world's hand starts or stops holding the duck up."""
        period = 1.0 / WORLD_HZ
        client: bodyhost.HostClient | None = None
        while not self._stop.is_set():
            try:
                if client is None:
                    client = bodyhost.HostClient(
                        int(self.body_port or 0), timeout_s=WORLD_TIMEOUT_S
                    )
                state = client.call("x_world")
                self.world_state = state
                self.world_at = time.monotonic()
                hand = state.get("hand") or {}
                self._set_held(bool(isinstance(hand, dict) and hand.get("held")))
            except Exception:
                if self._stop.is_set():
                    break
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                client = None
            self._stop.wait(period)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _set_held(self, held: bool) -> None:
        with self._lock:
            if held == self._held:
                return
            self._held = held
            if held:
                self._walk = self._drive = None
                self._posture_request = "sit"
            elif self._awake and self._wanted is not None:
                # Set down: back to what Duck Core wants (it may have asked while held).
                self._posture_request = self._wanted

    @property
    def robotd_socket(self) -> Path | None:
        """This body's ``robotd.sock`` (None before start)."""
        return None if self._state_dir is None else self._state_dir / "robotd.sock"

    @property
    def state_dir(self) -> Path | None:
        """This body's run folder (robotd's socket, logs; removed when the body stops)."""
        return self._state_dir

    def world(self) -> dict[str, Any] | None:
        """The world's latest truth (x_world; None before the first answer, without a world,
        or when it is older than MONITOR_STALE_S)."""
        if time.monotonic() - self.world_at > MONITOR_STALE_S:
            return None
        return self.world_state

    def world_call(self, op: str, **params: Any) -> dict[str, Any]:
        """One world op on the world server's port (``x_place``, ``x_hand``, ``x_people``, …),
        on a connection of its own. Raises RuntimeError with the world's words when it refuses
        (``{"error": ...}``) and ConnectionError when there is no world to ask."""
        if not self.home or self.body_port is None:
            raise ConnectionError("這個物理身體沒有物理世界（沒有房子、手和家人）")
        with self._world_lock:
            for attempt in (1, 2):
                sent = False
                try:
                    if self._world_client is None:
                        self._world_client = bodyhost.HostClient(
                            int(self.body_port), timeout_s=WORLD_TIMEOUT_S
                        )
                    self._world_client.send(op, **params)
                    sent = True
                    return self._world_client.receive(op)
                except (OSError, ValueError) as error:
                    client, self._world_client = self._world_client, None
                    try:
                        if client is not None:
                            client.close()
                    except Exception:
                        pass
                    # One fresh try only when the op cannot have run: nothing was sent, or a
                    # stale connection was closed or reset under it. A timeout or a garbled
                    # answer after sending may follow an op that ran (x_hand, x_place,
                    # x_roll_ball, a battery swap are not idempotent): never sent twice.
                    stale = not sent or isinstance(error, ConnectionError)
                    if attempt == 2 or not stale:
                        again = "" if not sent else "（已經送出，沒有再送一次）"
                        raise ConnectionError(f"world {op}: {error}{again}") from error
        raise ConnectionError(f"world {op}")  # (unreachable)

    def monitor_view(self) -> dict[str, Any]:
        """The latest stall monitor (see duckmind.bodyhost.assess), suspect when it is stale.

        Always every key of MONITOR_FIELDS: collision, lockstep, max_stall_ms_10s,
        max_stall_ms_60s, robotd_hz (robotd's writes in the last second), realtime (simulated
        per wall second over the last 10 s — the frame's own "realtime" is over ~2 s),
        trust ("ok" | "suspect") and why_zh (reasons in plain Traditional Chinese); bodyhost
        adds a few more (robotd_hz_min_10s, since_write_ms, stalls_10s, ...).
        """
        monitor = self.monitor
        age = time.monotonic() - self._monitor_at
        if math.isfinite(age) and age > MONITOR_STALE_S:
            why = [
                *monitor.get("why_zh", []),
                msg("livebody.LiveBody.monitor_view.270a224b", age=age),
            ]
            monitor = {**monitor, "trust": "suspect", "why_zh": why}
        return monitor

    def frame(self) -> dict[str, Any] | None:
        return self.latest

    def status(self) -> dict[str, Any]:
        return {
            "running": self.error is None and self.latest is not None,
            "policy": self.policy,
            "fallen": self.fallen,
            "realtime": round(self.realtime_factor, 3),
            "error": self.error,
            "monitor": self.monitor_view(),
            "held": self._held,
        }


def main(argv: list[str] | None = None) -> None:
    """Start the live body, play a few clips and a posture cycle, print what physics did."""
    parser = argparse.ArgumentParser(description="Official robotd + MuJoCo body smoke run")
    parser.add_argument("--microduck", type=Path)
    parser.add_argument("--microduck-rl", type=Path)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--robotd-bin", type=Path)
    parser.add_argument("--ort-lib", type=Path)
    parser.add_argument("--clips", default="moon_gaze,look_around,nod")
    args = parser.parse_args(argv)
    paths = LivePaths.from_env()
    if args.microduck:
        from duckmind.sim_live import _installed_ort_library

        ort = args.ort_lib or _installed_ort_library()
        paths = LivePaths(args.microduck, args.microduck_rl, args.policies, args.robotd_bin, ort)
    if paths is None:
        raise SystemExit("give the five paths or set the DUCKMIND_* variables")
    body = LiveBody(paths)
    started = time.monotonic()
    try:
        body.start()
        print(f"seated after {time.monotonic() - started:.1f}s: policy {body.policy}", flush=True)

        def report(label: str, seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                frame = body.frame() or {}
                joints = frame.get("joints", {})
                trunk = frame.get("trunk", [0, 0, 0])
                monitor = frame.get("monitor") or {}
                print(
                    f"{label:12s} rt={frame.get('realtime', 0):.2f} policy={frame.get('policy')} "
                    f"fallen={frame.get('fallen')} "
                    f"trunk=({trunk[0]:+.3f},{trunk[1]:+.3f},{trunk[2]:.3f}) "
                    + " ".join(f"{name[:6]}={joints.get(name, 0):+.2f}" for name in HEAD)
                    + f" trust={monitor.get('trust')} stall10s={monitor.get('max_stall_ms_10s')}"
                    f" robotd={monitor.get('robotd_hz')}",
                    flush=True,
                )
                time.sleep(0.5)

        for clip in args.clips.split(","):
            body.consume([{"clip": clip, "amplitude": 10_000, "tempo": 10_000}])
            report(clip, 4.0)
        body.consume([{"posture": "stand"}])
        report("stand", 4.0)
        body.consume([{"walk": {"vx": 0.06, "vyaw": 0.3, "seconds": 3.0}}])
        report("walk", 4.0)
        body.consume([{"posture": "sit"}])
        report("sit", 4.0)
    finally:
        body.stop()


if __name__ == "__main__":
    main()
