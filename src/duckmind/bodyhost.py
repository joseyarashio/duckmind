"""The official MuJoCo body server, hosted for the live simulator. No upstream file is edited.

``duckmind.livebody`` starts this instead of ``python -m mjlab_microduck.sim.body_server``
(same cwd = microduck_rl, PYTHONPATH = microduck_rl/src, same arguments) plus two flags::

    python -m duckmind.bodyhost --headless --keyframe SIT --port 7801 \
        [--collision allcollisions|groundcontact] [--lockstep | --no-lockstep]

It imports the official ``mjlab_microduck.sim.body_server`` and changes three things, none of
them a physics parameter (owner decisions 2026-09-26, phase 0「看見真相」):

1. **The collision model.** ``--collision allcollisions`` (the default) serves the official
   ``scene_allcollisions.xml``: the same scene as the official default ``scene.xml`` (floor,
   light, keyframes, options) with ``robot_allcollisions.xml`` included instead of
   ``robot_groundcontact.xml`` — 70 colliding robot geoms instead of 11, so thighs, knees and
   head meet the floor the way a real duck's do (phase 1: with groundcontact the thighs pass
   through the floor, so a walking fall needed a hand 4.7% of the time; with allcollisions
   33%). Masses, joints, actuators, timestep and floor contact are the same; the two official
   scenes are checked at start to differ only in the robot they include
   (:func:`check_scene_pair`).
2. **Lockstep stepping** (``--lockstep``, the default). The official server steps the world in
   its own real-time thread, 20 ms per pass, whatever robotd does. When the Mac is busy a pass
   is late and the next ones run back to back to catch up, with no robotd command in between —
   the only thing phase 1 found that makes a rise fail (a stall ≥ 60 ms: 78/96 clean rises;
   robotd < 45 Hz: 24/48; lockstep under the same stress: 64/64). Here that thread does not
   step: every ``write`` from robotd applies the joint targets and then advances the world by
   exactly one control period (4 × 5 ms) before it is answered, so robotd's read at tick k+1
   always sees the world 20 ms after its write at tick k. robotd times its policies in ticks
   (a fixed dt of 20 ms), so they stay in step with the physics; a busy Mac makes the
   simulation slower than the wall clock, never wrong. If robotd stops writing, the world
   stops with it — and the monitor says so.
3. **A stall monitor**, as extra ops on the body server's own port (robotd never sends them):
   ``x_monitor`` — the wall-clock gap between consecutive physics advances, robotd's write
   rate and the real-time factor, with a verdict (:func:`assess`) in plain Traditional
   Chinese — and ``x_info`` — what is loaded (scene, robot file, sha256s, lockstep).

Every official function this host patches, replaces or relies on (``World.step``,
``Handler.dispatch``, ``Handler.handle``, ``run``, ``main``) is pinned by the sha256 of its
source (:data:`UPSTREAM_SHA256`, the same hash ``inspect.getsource`` gives); if upstream
changed any of them the host refuses to start and names what changed. The rest of what
decides the physics — the whole body server file, the scenes, the robots and their meshes —
is pinned whole (:data:`REVIEWED_SHA256`): a change there starts, but the monitor keeps saying
the physics cannot be trusted, naming the file, until someone has reviewed it.

Nothing heavy is imported at import time: ``duckmind.livebody`` and the tests use
:class:`StallMonitor`, :func:`assess` and :class:`HostClient` without MuJoCo.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import json
import os
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NoReturn

from duckmind.i18n import msg

HOST_VERSION = "duckmind-bodyhost-1"

# collision model → (official scene file, the robot file it includes). Both in
# microduck_rl/src/mjlab_microduck/robot/microduck, referenced there, never copied.
COLLISIONS: dict[str, tuple[str, str]] = {
    "allcollisions": ("scene_allcollisions.xml", "robot_allcollisions.xml"),
    "groundcontact": ("scene.xml", "robot_groundcontact.xml"),
}
DEFAULT_COLLISION = "allcollisions"  # owner decision 2026-09-26
DEFAULT_LOCKSTEP = True  # owner decision 2026-09-26
CONTROL_PERIOD_S = 0.020  # robotd's 50 Hz tick; the official run() also steps 20 ms per pass
# The servo command delay (tr:gate 2026-09-28), opt-in: 0 = the joint targets act from the next
# physics step, as before. Training delays every target 15–30 ms (the velocity recipe's DR: the
# bus and the servo firmware), and the Stage-1 skill networks need it (Mac gates on a nominal
# duck: hop_back falls 30/30 with 0 ms, 0/30 with 20 ms; the shake is under its gate with 0 ms).
# Our live launcher sets it with the skills; the official server is not changed.
COMMAND_DELAY_ENV = "DUCKMIND_BODY_COMMAND_DELAY_MS"
COMMAND_DELAY_MAX_MS = 100.0

# The official functions this host patches, replaces or relies on, pinned by the sha256 of
# their source text as inspect.getsource returns it (checked 2026-09-27;
# phase1/body_launcher.py pinned World.step to the same hash on 2026-09-26).
UPSTREAM_SHA256: dict[str, str] = {
    # wrapped: every physics advance passes through it (the stall monitor's clock)
    "World.step": "f3cd6759f843067d68f66a62f8fc1ee452e386fd1f0c14b737cda10411532533",
    # wrapped: the x_ ops, and in lockstep one control period of physics after each write
    "Handler.dispatch": "2f2d1a37193ab9aeaafb575747aa8d29f7da95c334b558b55594322a502b555d",
    # relied on: it answers a request only after dispatch returns (lockstep ordering)
    "Handler.handle": "5e0dd121c36f5c3193e88a10dd6d3d56ecd3f6d56db8dfe8f960122af48458ed",
    # replaced in lockstep: the real-time stepping loop (+ viewer and cameras, refused here)
    "run": "8e16fd9fdf17ffdfa62ea1cd7a689b95edf9782fc511f3420d438dfe9ba55505",
    # called: builds the world from --scene/--keyframe and hands it to run()
    "main": "bea01a8ae99ba911889458a14f995f78110c97ccfe4dc02e2c1e5ec1a5eca8e3",
    # wrapped: the servos a scripted pose hangs free (torque_joints) keep zero gain
    "Body._apply_torque": "bcd6ef71cc45d04f5f4f944bdba2c2b88e0edb50bda1d60647e96744a18e3434",
}

# Dynamixel ids in wire order (duck_control::model::JOINT_IDS): the patched robotd's
# torque_joints op names servos by id, as on the bus.
SERVO_IDS: tuple[int, ...] = (20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 10, 11, 12, 13, 14)

# Everything else that decides the physics, pinned whole (reviewed 2026-09-27): the official
# body server file (TIMESTEP, HOME_POSE, Body.set_targets, Body.sensors, build_world, ...), the
# scene and robot files of both collision models, and the meshes the robots use (one digest
# over their names and contents, see mesh_digest). A change here does not stop the host — the
# functions above are what it relies on — but until someone has reviewed it and updated this
# table the monitor says the physics may have changed (trust suspect, the file named).
REVIEWED_SHA256: dict[str, str] = {
    "body_server.py": "2040c827628c33fee998944bce1fcd456c4b993802bb473f016681688bff7b08",
    "scene_allcollisions.xml": "0d2f58227a3e1c4caae4e847a368ae83516d0f398f428c4e96737b67dd894ef1",
    "robot_allcollisions.xml": "ec9d2f731596cfc847a7fc3898ef8b5e67a548928abc23216537f1da8af2729c",
    "scene.xml": "9a85461121ba8273083c482c166f27e143ca386993b3ec8f8d97eb09aa257d31",
    "robot_groundcontact.xml": "99d23867e46e9d4748ff1398016b850f09fb6cf02a558c0272743b39ce3acf15",
    "meshes": "b9c510c119cd1ab9a4e9008b033c69d0e8cf1031a1a90e3f58ed5669ba16aa21",
}

# When the physics stops being trustworthy (phase 1, report_zh.md §3 and rise.md §3.8):
# rises failed only after a physics stall of 60 ms or more (78/96 clean), with robotd below
# 45 ticks a second (24/48; also robotd's own health gate), or with the simulation behind
# the wall clock; with the physics keeping up, 141/141.
STALL_LIMIT_MS = 60.0
# In lockstep a stall is a late robotd tick: the world waits for it, and robotd's next read
# still sees exactly 20 ms of physics after its write, so the catch-up that made rises fail
# cannot happen (phase 1: lockstep with 10 % × 100 ms stalls 30/30, with robotd slowed to
# 17-20 Hz 30/30). What robotd cannot ride out is an answer later than its own simulator
# timeout (duck-control sim.rs TIMEOUT, 200 ms: it drops the link and reconnects) — that is
# the lockstep limit. Shorter stalls are only reported in the numbers; a slow second or a
# simulation behind the wall clock is still caught by the robotd and real-time rules below.
LOCKSTEP_STALL_LIMIT_MS = 200.0
ROBOTD_MIN_HZ = 45
REALTIME_MIN = 0.97
WINDOW_S = 10.0  # the verdict looks this far back
LONG_WINDOW_S = 60.0  # also reported: the longest stall in the last minute
SILENT_S = 0.5  # no write for this long: robotd has stopped, not just stalled
MIN_SPAN_S = 1.0  # the real-time factor needs at least this much history


# ── the guards ──────────────────────────────────────────────────────────────────


def _definition(tree: ast.Module, qualname: str) -> ast.AST | None:
    nodes: list[ast.stmt] = tree.body
    found: ast.AST | None = None
    for part in qualname.split("."):
        found = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == part
            ),
            None,
        )
        if found is None:
            return None
        nodes = getattr(found, "body", [])
    return found


def source_digests(text: str, names: Iterable[str]) -> dict[str, str | None]:
    """sha256 of each named function's source, exactly the text ``inspect.getsource`` returns
    for it (its lines, indentation and decorators included); None when it is not there."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    digests: dict[str, str | None] = {}
    for name in names:
        node = _definition(tree, name)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            digests[name] = None
            continue
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
        source = "".join(lines[start - 1 : node.end_lineno])
        digests[name] = hashlib.sha256(source.encode()).hexdigest()
    return digests


def check_upstream(path: Path, pinned: dict[str, str] | None = None) -> list[str]:
    """One line per pinned official function whose source changed (empty: all as reviewed)."""
    pinned = UPSTREAM_SHA256 if pinned is None else pinned
    found = source_digests(path.read_text(encoding="utf-8"), pinned)
    problems = []
    for name, want in pinned.items():
        got = found.get(name)
        if got is None:
            problems.append(f"{name}：找不到這個函式")
        elif got != want:
            problems.append(f"{name}：sha256 {got[:12]}…，核對過的是 {want[:12]}…")
    return problems


def mesh_digest(robot: Path) -> str | None:
    """One sha256 over the mesh files a robot file uses: each name and its content's sha256,
    in name order (None when the robot file or one of its meshes is missing)."""
    try:
        root = ET.parse(robot).getroot()
    except (OSError, ET.ParseError):
        return None
    compiler = root.find("compiler")
    folder = robot.parent / ((compiler.get("meshdir") if compiler is not None else None) or "")
    digest = hashlib.sha256()
    for name in sorted({str(node.get("file")) for node in root.iter("mesh") if node.get("file")}):
        path = folder / name
        if not path.is_file():
            return None
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def check_reviewed(
    body_server: Path, mjcf_dir: Path, collision: str, pinned: dict[str, str] | None = None
) -> list[str]:
    """One plain line per physics file this collision model uses that is not the reviewed one
    (:data:`REVIEWED_SHA256`); empty when everything is as reviewed."""
    pinned = REVIEWED_SHA256 if pinned is None else pinned
    scene_name, robot_name = COLLISIONS[collision]
    found = {
        "body_server.py": _file_sha256(body_server),
        scene_name: _file_sha256(mjcf_dir / scene_name),
        robot_name: _file_sha256(mjcf_dir / robot_name),
        "meshes": mesh_digest(mjcf_dir / robot_name),
    }
    names = {"meshes": f"{robot_name} 用的網格（{mjcf_dir.name}/assets）"}
    return [
        f"上游的 {names.get(key, key)} 和核對過的版本不一樣：還沒有人核對，物理可能變了"
        for key, got in found.items()
        if got is None or got != pinned.get(key)
    ]


def _scene_signature(text: str, robot_file: str) -> str:
    """The scene with its robot include blanked out, comments and layout dropped."""
    root = ET.fromstring(text)
    for node in root.iter("include"):
        if node.get("file") == robot_file:
            node.set("file", "ROBOT")
    return ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)


def check_scene_pair(mjcf_dir: Path) -> list[str]:
    """Differences between the official scenes of the two collision models other than the
    robot file each includes (empty: choosing one changes the collision model and nothing
    else in the scene: floor, light, keyframes, options)."""
    problems: list[str] = []
    signatures: dict[str, str] = {}
    for collision, (scene_name, robot_name) in COLLISIONS.items():
        scene = mjcf_dir / scene_name
        if not scene.is_file():
            problems.append(f"{scene_name}：找不到")
            continue
        if not (mjcf_dir / robot_name).is_file():
            problems.append(f"{robot_name}：找不到")
            continue
        text = scene.read_text(encoding="utf-8")
        try:
            includes = [node.get("file") for node in ET.fromstring(text).iter("include")]
        except ET.ParseError as error:
            problems.append(f"{scene_name}：讀不懂（{error}）")
            continue
        if includes != [robot_name]:
            problems.append(f"{scene_name}：引入的是 {includes}，應該只有 {robot_name}")
            continue
        signatures[collision] = _scene_signature(text, robot_name)
    if len(signatures) == len(COLLISIONS) and len(set(signatures.values())) > 1:
        first, second = (signatures[name] for name in COLLISIONS)
        at = next(
            (i for i, (a, b) in enumerate(zip(first, second)) if a != b),
            min(len(first), len(second)),
        )
        names = " 和 ".join(scene for scene, _ in COLLISIONS.values())
        problems.append(f"{names} 除了機器人檔以外還有不同：…{first[max(0, at - 40) : at + 40]}…")
    return problems


def scene_for(mjcf_dir: Path, collision: str) -> Path:
    """The official scene file that serves this collision model."""
    if collision not in COLLISIONS:
        raise ValueError(f"unknown collision model {collision!r}; known: {sorted(COLLISIONS)}")
    return mjcf_dir / COLLISIONS[collision][0]


def command_delay_steps(timestep: float, environ: dict[str, str] | None = None) -> int:
    """Physics steps a joint target waits before the servos act on it (``$DUCKMIND_BODY_
    COMMAND_DELAY_MS`` rounded to whole ``timestep``s; unset or 0 → 0)."""
    raw = ((os.environ if environ is None else environ).get(COMMAND_DELAY_ENV) or "").strip()
    if not raw:
        return 0
    ms = float(raw)
    if not (0.0 <= ms <= COMMAND_DELAY_MAX_MS):
        raise ValueError(f"{COMMAND_DELAY_ENV}={raw}: 0 to {COMMAND_DELAY_MAX_MS:g} ms")
    return int(round(ms / 1000.0 / timestep))


def steps_per_write(timestep: float) -> int:
    """Physics steps in one control period — the same batch the official run() takes."""
    return max(1, round(CONTROL_PERIOD_S / timestep))


ROBOT_ROOT_BODY = "trunk_base"  # the duck's free-floating root body in the official robots


def robot_root(model: Any) -> int | None:
    """The duck's root body id (``trunk_base``; body 1 when names cannot be read), or None in
    a model with no bodies but the world."""
    nbody = int(model.nbody)
    try:
        found = int(model.body(ROBOT_ROOT_BODY).id)
    except Exception:  # an unnamed stand-in model
        found = 1
    return found if 0 < found < nbody else None


def robot_colliding_geoms(model: Any) -> int:
    """Colliding geoms of the duck itself: its root body's whole tree, never the room's
    toys or other free bodies (which are not on the world body either)."""
    root = robot_root(model)
    if root is None:
        return 0
    return sum(
        1
        for geom in range(int(model.ngeom))
        if int(model.body_rootid[int(model.geom_bodyid[geom])]) == root
        and (int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))
    )


HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft", "mandible")  # the head chain
BEAK_BODIES = (
    "jaw_soft",
    "mandible",
)  # … and the beak (the composed model adds the mandible, world/jaw.py)


def mine_set(ids: Any) -> frozenset[int]:
    """The body ids of the duck as a set."""
    return frozenset(int(i) for i in ids)


class ContactWatch:
    """Counts the physics steps (one per robotd write) in which the duck's head chain or its beak touches
    something (``x_contacts``: the emotions' check, 2026-10-08). ``head_world``: the head against the floor, a wall or a
    thing; ``head_trunk``: against the rest of the duck (trunk, legs); ``beak_world`` / ``beak_trunk`` the same for the
    beak alone, and ``beak_neck`` the beak against its own head's other parts (the lower beak on the neck housing)."""

    KEYS = ("head_world", "head_trunk", "beak_world", "beak_trunk", "beak_neck")

    def __init__(self) -> None:
        self.writes = 0
        self.counts = dict.fromkeys(self.KEYS, 0)
        self.pairs: dict[str, int] = {}
        self._ids: tuple[Any, ...] | None = None
        # the impact numbers (play dead, 2026-10-11), taken at every physics step while this watch runs: the largest
        # change of the trunk's velocity in one step as an acceleration (m/s^2), its fastest turn (rad/s) and the largest
        # contact force between the duck and anything else (N) — what a lie-down is compared with a plain sit by
        self.peak_acc = 0.0
        self.peak_gyro = 0.0
        self.peak_force = 0.0
        self._v: Any = None

    def substep(self, world: Any) -> None:
        """After one ``mj_step``: the impact numbers of this step."""
        import mujoco
        import numpy as np

        model, data = world.model, world.data
        _head, _beak, mine = self._resolve(model)
        root = robot_root(model)
        with world.lock:
            if root is not None and int(model.body_dofnum[root]) >= 6:
                adr = int(model.body_dofadr[root])
                vel = np.array(data.qvel[adr : adr + 3])
                gyro = float(np.linalg.norm(data.qvel[adr + 3 : adr + 6]))
                if self._v is not None:
                    self.peak_acc = max(
                        self.peak_acc,
                        float(np.linalg.norm(vel - self._v)) / float(model.opt.timestep),
                    )
                self._v = vel
                self.peak_gyro = max(self.peak_gyro, gyro)
            force = np.zeros(6)
            duck = mine_set(mine)
            for i in range(int(data.ncon)):
                c = data.contact[i]
                in1 = int(model.geom_bodyid[c.geom1]) in duck
                in2 = int(model.geom_bodyid[c.geom2]) in duck
                if in1 == in2:
                    continue  # (the duck against itself, or two things of the world)
                mujoco.mj_contactForce(model, data, i, force)
                self.peak_force = max(self.peak_force, float(force[0]))

    def _resolve(self, model: Any) -> tuple[Any, ...]:
        import numpy as np

        if self._ids is None:

            def ids(names: tuple[str, ...]) -> Any:
                out = []
                for name in names:
                    try:
                        out.append(int(model.body(name).id))
                    except Exception:  # (a model without that body)
                        continue
                return np.array(sorted(out), dtype=int)

            root = robot_root(model)
            mine = np.array(
                [
                    i
                    for i in range(int(model.nbody))
                    if root is not None and int(model.body_rootid[i]) == root
                ],
                dtype=int,
            )
            self._ids = (ids(HEAD_BODIES), ids(BEAK_BODIES), mine)
        return self._ids

    def scan(self, world: Any) -> None:
        import numpy as np

        model, data = world.model, world.data
        head, beak, mine = self._resolve(model)
        self.writes += 1
        with world.lock:
            n = int(data.ncon)
            if n == 0:
                return
            b1 = np.array(model.geom_bodyid[data.contact.geom1[:n]])
            b2 = np.array(model.geom_bodyid[data.contact.geom2[:n]])
        seen: set[str] = set()
        for a, b in ((b1, b2), (b2, b1)):
            on_head = np.isin(a, head)
            if not on_head.any():
                continue
            other = b[on_head]
            mine_other = np.isin(other, mine)
            in_head = np.isin(other, head)
            if (~mine_other).any():
                seen.add("head_world")
            if (mine_other & ~in_head).any():
                seen.add("head_trunk")
            on_beak = np.isin(a, beak)
            if on_beak.any():
                ob = b[on_beak]
                if (~np.isin(ob, mine)).any():
                    seen.add("beak_world")
                if (np.isin(ob, mine) & ~np.isin(ob, head)).any():
                    seen.add("beak_trunk")
                if (np.isin(ob, head) & ~np.isin(ob, beak)).any():
                    seen.add("beak_neck")
            for x, y in zip(a[on_head], other, strict=True):
                if int(x) in set(beak.tolist()) and int(y) in set(beak.tolist()):
                    continue
                if int(x) not in set(beak.tolist()) and int(y) in set(head.tolist()):
                    continue
                key = f"{model.body(int(x)).name}|{model.body(int(y)).name}"
                self.pairs[key] = self.pairs.get(key, 0) + 1
        for name in seen:
            self.counts[name] += 1

    def read(self) -> dict[str, Any]:
        top = sorted(self.pairs.items(), key=lambda kv: -kv[1])[:8]
        return {
            "writes": self.writes,
            **self.counts,
            "pairs": dict(top),
            "peak_acc": round(self.peak_acc, 2),
            "peak_gyro": round(self.peak_gyro, 2),
            "peak_force": round(self.peak_force, 3),
        }


# ── the monitor ─────────────────────────────────────────────────────────────────


class StallMonitor:
    """When the physics advanced and when robotd wrote, on the wall clock, and what that adds
    up to. Thread-safe: ``advanced`` and ``wrote`` come from whichever thread steps (the
    official real-time loop, or robotd's connection in lockstep), ``stats`` from anyone."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        keep_s: float = LONG_WINDOW_S + WINDOW_S,
    ) -> None:
        self.clock = clock
        self.keep_s = keep_s
        self._lock = threading.Lock()
        # (wall, simulated time after it, wall gap since the advance before)
        self._advances: deque[tuple[float, float, float]] = deque()
        self._writes: deque[float] = deque()
        self._last_advance: tuple[float, float] | None = None
        self._first_write: float | None = None
        self._last_write: float | None = None
        self.advance_count = 0
        self.write_count = 0
        self.write_errors = 0

    def advanced(self, sim_time: float, wall: float | None = None) -> None:
        """The world just moved on (one batch of steps); ``sim_time`` is where it is now."""
        now = self.clock() if wall is None else wall
        with self._lock:
            gap = now - self._last_advance[0] if self._last_advance is not None else 0.0
            self._advances.append((now, float(sim_time), gap))
            self._last_advance = (now, float(sim_time))
            self.advance_count += 1
            self._prune(now)

    def wrote(self, ok: bool = True, wall: float | None = None) -> None:
        """robotd sent joint targets (``ok``: the server took them)."""
        now = self.clock() if wall is None else wall
        with self._lock:
            self._writes.append(now)
            if self._first_write is None:
                self._first_write = now
            self._last_write = now
            self.write_count += 1
            if not ok:
                self.write_errors += 1
            self._prune(now)

    def _prune(self, now: float) -> None:
        horizon = now - self.keep_s
        # The advance at the start of the long window is kept: it anchors the real-time factor.
        while len(self._advances) > 1 and self._advances[1][0] <= horizon:
            self._advances.popleft()
        while self._writes and self._writes[0] < horizon:
            self._writes.popleft()

    def stats(self, now: float | None = None) -> dict[str, Any]:
        """Numbers over the last :data:`WINDOW_S` and :data:`LONG_WINDOW_S` seconds.

        ``max_stall_ms_*`` is the longest wall-clock gap between two physics advances that ended
        in the window, or the gap still open now (a world that stopped is a stall that has not
        ended). ``robotd_hz`` is robotd's writes in the last second (what robotd's own
        ``loop.hz`` counts), ``robotd_hz_min_10s`` the fewest in any whole second of the last
        ten since it started writing. ``realtime`` is simulated seconds per wall second over
        the last ten (None before a second of history).
        """
        now = self.clock() if now is None else now
        with self._lock:
            advances = list(self._advances)
            writes = list(self._writes)
            last = self._last_advance
            first_write, last_write = self._first_write, self._last_write
            counts = (self.advance_count, self.write_count, self.write_errors)

        def longest(window: float) -> float | None:
            since = now - window
            gaps = [gap for wall, _, gap in advances if wall >= since]
            if last is not None:
                gaps.append(now - last[0])
            return max(gaps) * 1000.0 if gaps else None

        def stalls(window: float) -> int:
            since = now - window
            limit = STALL_LIMIT_MS / 1000.0
            open_gap = last is not None and now - last[0] >= limit
            return sum(1 for wall, _, gap in advances if wall >= since and gap >= limit) + open_gap

        def written(start: float, end: float) -> int:
            return bisect.bisect_right(writes, end) - bisect.bisect_right(writes, start)

        rates = [
            written(now - k - 1.0, now - k)
            for k in range(int(WINDOW_S))
            if first_write is not None and now - k - 1.0 >= first_write
        ]
        realtime = None
        if last is not None and advances:
            walls = [wall for wall, _, _ in advances]
            index = bisect.bisect_right(walls, now - WINDOW_S) - 1
            ref_wall, ref_sim, _ = advances[max(0, index)]
            span = now - ref_wall
            if span >= MIN_SPAN_S:
                realtime = (last[1] - ref_sim) / span
        return {
            "max_stall_ms_10s": _round(longest(WINDOW_S), 1),
            "max_stall_ms_60s": _round(longest(LONG_WINDOW_S), 1),
            "stalls_10s": stalls(WINDOW_S),
            "stalls_60s": stalls(LONG_WINDOW_S),
            "robotd_hz": written(now - 1.0, now) if first_write is not None else None,
            "robotd_hz_min_10s": min(rates) if rates else None,
            "realtime": _round(realtime, 3),
            "since_write_ms": _round(None if last_write is None else (now - last_write) * 1000, 1),
            "since_advance_ms": _round(None if last is None else (now - last[0]) * 1000, 1),
            "sim_time": None if last is None else round(last[1], 4),
            "advances": counts[0],
            "writes": counts[1],
            "write_errors": counts[2],
        }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def assess(stats: dict[str, Any]) -> tuple[str, list[str]]:
    """("ok" | "suspect", the reasons in plain Traditional Chinese) for monitor numbers.

    Suspect when robotd has stopped (or not started) writing, a physics stall of
    :data:`STALL_LIMIT_MS` or more happened in the last ten seconds (in lockstep
    :data:`LOCKSTEP_STALL_LIMIT_MS`), robotd fell below :data:`ROBOTD_MIN_HZ` in any second of
    them, or the simulation ran slower than :data:`REALTIME_MIN` of the wall clock over them —
    what phase 1 found rises fail under. robotd is named once, as the duck's control program.
    """
    lockstep = bool(stats.get("lockstep"))
    why: list[str] = []
    silent = stats.get("since_write_ms")
    if silent is None:
        why.append(
            msg(
                "bodyhost.assess.32180b23",
                choice=msg("bodyhost.assess.b53e23ae") if lockstep else "",
            )
        )
        return "suspect", why
    if silent >= SILENT_S * 1000.0:
        seconds = silent / 1000.0
        if lockstep:
            why.append(msg("bodyhost.assess.07fced96", seconds=seconds))
        else:
            why.append(msg("bodyhost.assess.47a6c2e8", seconds=seconds))
        return "suspect", why
    stall = stats.get("max_stall_ms_10s")
    if lockstep and stall is not None and stall >= LOCKSTEP_STALL_LIMIT_MS:
        why.append(msg("bodyhost.assess.6bb2dde9", stall=stall))
    elif not lockstep and stall is not None and stall >= STALL_LIMIT_MS:
        why.append(msg("bodyhost.assess.a17e22b8", stall=stall))
    slowest = stats.get("robotd_hz_min_10s")
    if slowest is not None and slowest < ROBOTD_MIN_HZ:
        why.append(msg("bodyhost.assess.15e26cab", slowest=slowest, ROBOTD_MIN_HZ=ROBOTD_MIN_HZ))
    realtime = stats.get("realtime")
    if realtime is not None and realtime < REALTIME_MIN:
        cause = msg("bodyhost.assess.1428e8f9") if lockstep else msg("bodyhost.assess.2cb0db19")
        why.append(msg("bodyhost.assess.b03bfce0", realtime=realtime, cause=cause))
    return ("suspect" if why else "ok"), why


# ── the host ────────────────────────────────────────────────────────────────────


class Host:
    """What this process adds to the official body server: the monitor, lockstep and x_ ops."""

    def __init__(
        self,
        *,
        collision: str,
        lockstep: bool,
        scene: Path,
        monitor: StallMonitor | None = None,
    ) -> None:
        if collision not in COLLISIONS:
            raise ValueError(f"unknown collision model {collision!r}")
        self.collision = collision
        self.lockstep = lockstep
        self.scene = scene
        self.monitor = monitor or StallMonitor()
        self.upstream: Path | None = None
        # Physics files that differ from the reviewed ones (check_reviewed): each is a reason
        # the monitor gives for not trusting the physics until someone has reviewed it.
        self.unreviewed: list[str] = []
        # The servo command delay (COMMAND_DELAY_ENV), in physics steps; the targets in flight
        self.contacts: ContactWatch | None = (
            None  # x_contacts start: the head's contacts since then
        )
        self.command_delay_ms = float((os.environ.get(COMMAND_DELAY_ENV) or "0").strip() or 0)
        self.command_delay: int | None = None  # steps; resolved on the first write (timestep)
        self._in_flight: deque[Any] = deque()

    def install(self, bs: Any) -> None:
        """Patch the official module in place (only after :func:`check_upstream` passed)."""
        original_step = bs.World.step
        original_dispatch = bs.Handler.dispatch
        original_apply_torque = bs.Body._apply_torque
        host = self

        def step(world: Any, times: int = 1) -> None:
            watch = host.contacts
            if watch is None:
                original_step(world, times)
            else:  # (the contact watch measures the impact of every physics step, not of every control period)
                for _ in range(times):
                    original_step(world, 1)
                    watch.substep(world)
            host.monitor.advanced(float(world.data.time))

        def apply_torque(body: Any) -> None:
            # The servos a scripted pose hangs free (torque_joints) stay at zero gain whatever
            # the rest are set to; a full torque on (robot.init) clears them.
            original_apply_torque(body)
            off = getattr(body, "_x_torque_off", None)
            if off and body.torque_on:
                slots = [slot for slot, wire in enumerate(body.to_wire) if wire in off]
                if slots:
                    index = body.actuator_slice[slots]
                    model = body.world.model
                    model.actuator_gainprm[index, 0] = 0.0
                    model.actuator_biasprm[index, 1] = 0.0

        def dispatch(handler: Any, body: Any, request: dict[str, Any]) -> dict[str, Any]:
            op = request.get("op")
            if isinstance(op, str) and op.startswith("x_"):
                return host.extra(op, body, request)
            if op == "torque_joints":
                return host.torque_joints(body, request)
            if op == "torque":
                # torque on sets the targets to where the joints are: nothing stale in flight
                host._in_flight.clear()
                if request.get("on"):
                    body._x_torque_off = set()
            if op != "write":
                return original_dispatch(handler, body, request)
            held = host.held_targets(body) if host.lockstep else None
            ok = False
            try:
                answer = original_dispatch(handler, body, request)
                ok = True
                return answer
            finally:
                # A tick happened even when its targets were refused: the servos keep the old
                # ones while the world moves on, as a failed bus write does on a robot.
                host.monitor.wrote(ok)
                if host.lockstep:
                    world = body.world
                    steps = steps_per_write(float(world.model.opt.timestep))
                    if host.delay_steps(world) > 0:
                        host.step_delayed(body, world, steps, held)
                    else:
                        world.step(steps)
                    if host.contacts is not None:
                        host.contacts.scan(world)

        bs.World.step = step
        bs.Handler.dispatch = dispatch
        bs.Body._apply_torque = apply_torque
        if self.lockstep:
            bs.run = idle_run
        self.upstream = Path(bs.__file__)

    def torque_joints(self, body: Any, request: dict[str, Any]) -> dict[str, Any]:
        """Torque on or off for some servos only: the sim side of the patched robotd's
        ``robot.poseJoints`` (play dead hangs the head servos free). ``ids`` are servo ids."""
        try:
            ids = [int(i) for i in request.get("ids") or []]
            on = bool(request.get("on"))
        except (TypeError, ValueError):
            return {"error": "torque_joints wants ids (servo ids) and on"}
        wires = {SERVO_IDS.index(i) for i in ids if i in SERVO_IDS}
        with body.world.lock:
            off = set(getattr(body, "_x_torque_off", set()))
            body._x_torque_off = (off - wires) if on else (off | wires)
            body._apply_torque()
        return {}

    def delay_steps(self, world: Any) -> int:
        if self.command_delay is None:
            timestep = float(world.model.opt.timestep)
            self.command_delay = command_delay_steps(
                timestep, {COMMAND_DELAY_ENV: repr(self.command_delay_ms)}
            )
            self._in_flight = deque(maxlen=self.command_delay + 1)
        return self.command_delay

    def held_targets(self, body: Any) -> Any:
        """The targets the servos hold before this write (None without a delay)."""
        if not self.command_delay_ms:
            return None
        with body.world.lock:
            return body.world.data.ctrl[body.actuator_slice].copy()

    def step_delayed(self, body: Any, world: Any, steps: int, held: Any) -> None:
        """Advance ``steps`` physics steps with the servos acting on the targets written
        :attr:`command_delay` steps earlier (the harness's and training's command delay)."""
        index = body.actuator_slice
        with world.lock:
            commanded = world.data.ctrl[index].copy()
        if not self._in_flight:
            start = held if held is not None else commanded
            for _ in range(int(self.command_delay or 0)):
                self._in_flight.append(start)
        for _ in range(steps):
            self._in_flight.append(commanded)
            with world.lock:
                world.data.ctrl[index] = self._in_flight[0]
            world.step(1)
        with world.lock:
            # between writes the servos keep acting on the target of this step (the world does
            # not move until robotd's next write in lockstep)
            world.data.ctrl[index] = self._in_flight[0]

    def extra(self, op: str, body: Any, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """The x_ ops: for the simulator and its evaluators only; robotd never sends them."""
        if op == "x_monitor":
            return self.report()
        if op == "x_info":
            return self.info(body)
        if op == "x_contacts":
            return self.contact_op(body, str((request or {}).get("do") or "read"))
        raise ValueError(f"unknown op {op!r}")

    def contact_op(self, body: Any, do: str = "read") -> dict[str, Any]:
        """``x_contacts``: ``start`` (count from now), ``read`` (the counts so far), ``stop``."""
        if do == "start":
            self.contacts = ContactWatch()
        elif do == "stop":
            self.contacts = None
        return {
            "ok": True,
            **(self.contacts.read() if self.contacts is not None else {"writes": 0}),
        }

    def report(self) -> dict[str, Any]:
        stats = {"collision": self.collision, "lockstep": self.lockstep, **self.monitor.stats()}
        trust, why = assess(stats)
        if self.unreviewed:
            trust, why = "suspect", [*why, *self.unreviewed]
        return {**stats, "trust": trust, "why_zh": why}

    def info(self, body: Any) -> dict[str, Any]:
        world = body.world
        model = world.model
        with world.lock:
            timestep = float(model.opt.timestep)
            colliding = robot_colliding_geoms(model)
            root = robot_root(model)
            mass = float(model.body_subtreemass[root]) if root is not None else None
        robot = self.scene.parent / COLLISIONS[self.collision][1]
        return {
            "host": HOST_VERSION,
            "collision": self.collision,
            "lockstep": self.lockstep,
            "command_delay_ms": self.command_delay_ms,
            "command_delay_steps": self.command_delay,
            "steps_per_write": steps_per_write(timestep),
            "timestep": timestep,
            "scene": str(self.scene),
            "scene_sha256": _file_sha256(self.scene),
            "robot_file": robot.name,
            "robot_sha256": _file_sha256(robot),
            "robot_colliding_geoms": colliding,
            "robot_mass_kg": None if mass is None else round(mass, 4),
            "body_server": str(self.upstream),
            "body_server_sha256": _file_sha256(self.upstream) if self.upstream else None,
            "upstream_sha256": dict(UPSTREAM_SHA256),
            "unreviewed": list(self.unreviewed),
            "mujoco": _mujoco_version(),
            "pid": os.getpid(),
            "argv": sys.argv[1:],
        }


def idle_run(world: Any, headless: bool) -> None:
    """The official ``run()`` minus its stepping: in lockstep only robotd's writes move the
    world. (Its other duties, a viewer and head cameras, are refused in lockstep.)"""
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mujoco_version() -> str | None:
    module = sys.modules.get("mujoco")
    return getattr(module, "__version__", None)


class HostClient:
    """A connection to the host's x_ ops (``x_monitor``, ``x_info``), apart from robotd's."""

    def __init__(self, port: int, *, timeout_s: float = 2.0) -> None:
        self._socket = socket.create_connection(("127.0.0.1", port), timeout=timeout_s)
        self._socket.settimeout(timeout_s)
        self._stream = self._socket.makefile("rwb")

    def call(self, op: str, **params: Any) -> dict[str, Any]:
        """One op and its answer (``params`` become the request's other fields)."""
        self.send(op, **params)
        return self.receive(op)

    def send(self, op: str, **params: Any) -> None:
        """Write one request (the first half of :meth:`call`)."""
        self._stream.write((json.dumps({**params, "op": op}) + "\n").encode())
        self._stream.flush()

    def receive(self, op: str) -> dict[str, Any]:
        """Read the answer to the request just sent (the second half of :meth:`call`)."""
        line = self._stream.readline()
        if not line:
            raise ConnectionError("the body host closed the connection")
        answer = json.loads(line)
        if not isinstance(answer, dict) or "error" in answer:
            raise RuntimeError(f"body host {op}: {answer}")
        return answer

    def monitor(self) -> dict[str, Any]:
        return self.call("x_monitor")

    def info(self) -> dict[str, Any]:
        return self.call("x_info")

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._socket.close()


# ── the process ─────────────────────────────────────────────────────────────────


def _refuse(headline: str, lines: list[str]) -> NoReturn:
    sys.stderr.write(
        "== duckmind bodyhost 拒絕啟動：" + headline + "\n" + "".join(f"   {x}\n" for x in lines)
    )
    sys.stderr.flush()
    raise SystemExit(3)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="The official MuJoCo body server with full-body collisions, lockstep "
        "stepping and a stall monitor. Other arguments go to the official server.",
        allow_abbrev=False,  # never take one of the official server's options by its prefix
    )
    parser.add_argument("--collision", choices=sorted(COLLISIONS), default=DEFAULT_COLLISION)
    lock = parser.add_mutually_exclusive_group()
    lock.add_argument("--lockstep", dest="lockstep", action="store_true")
    lock.add_argument("--no-lockstep", dest="lockstep", action="store_false")
    parser.set_defaults(lockstep=DEFAULT_LOCKSTEP)
    ours, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--scene")
    peek.add_argument("--ducks", type=int, default=1)
    peek.add_argument("--cameras", default="")
    peek.add_argument("--headless", action="store_true")
    seen, _ = peek.parse_known_args(rest)
    if seen.scene is not None:
        _refuse("--scene 由 --collision 決定", ["用 --collision allcollisions 或 groundcontact"])
    if ours.lockstep and (seen.ducks != 1 or seen.cameras.strip() or not seen.headless):
        _refuse(
            "lockstep 只支援一隻鴨、--headless、沒有頭部鏡頭",
            ["每一次 robotd 寫入都讓整個世界前進 20 ms；要多隻鴨、畫面或鏡頭就加 --no-lockstep"],
        )

    from mjlab_microduck.sim import body_server as bs  # cwd/PYTHONPATH as livebody sets them

    upstream = Path(bs.__file__).resolve()
    changed = check_upstream(upstream)
    if changed:
        _refuse(
            f"上游的身體伺服器 {upstream} 的程式和核對過的不一樣，為了不讓物理悄悄改變，先不啟動",
            [
                *changed,
                "請先確認 src/duckmind/bodyhost.py 對新版還正確，再更新 UPSTREAM_SHA256",
            ],
        )
    mjcf = Path(getattr(bs, "SCENES", upstream.parents[1] / "robot" / "microduck"))
    different = check_scene_pair(mjcf)
    if different:
        _refuse("上游的兩個場景不再是「只差碰撞模型」，先不啟動", different)
    scene = scene_for(mjcf, ours.collision)
    host = Host(collision=ours.collision, lockstep=ours.lockstep, scene=scene)
    host.unreviewed = check_reviewed(upstream, mjcf, ours.collision)
    for line in host.unreviewed:
        print(f"== {HOST_VERSION}: {line}", flush=True)
    host.install(bs)
    sys.argv = [sys.argv[0], *rest, "--scene", str(scene)]
    print(
        f"== {HOST_VERSION}: {ours.collision} ({COLLISIONS[ours.collision][1]}), "
        f"lockstep {'on' if ours.lockstep else 'off'}, monitor on x_monitor",
        flush=True,
    )
    bs.main()


if __name__ == "__main__":
    main()
