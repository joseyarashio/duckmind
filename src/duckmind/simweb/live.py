"""Optional live physics for the simulator page: the official robotd and MuJoCo body, in a home.

When the simulator starts with ``--live``, every motion Duck Core produces (the same
motion dicts the page's feed carries) is also played on a real ``robotd --sim`` through
:class:`duckmind.livebody.LiveBody`, and the page draws what the physics did instead of
an idealized animation. Everything here is best effort: if the official pieces are
missing or the body falls over, the page keeps working and says why.

Phase 0 (「看見真相」) made the solid duck on the page the physics truth only. Phase 2
(「真的家與真的感測器」, physical-design spec §17; owner decisions A1–F5) puts that body in a
real home: the world server (:mod:`duckmind.world.simhome`) runs the page's 6×6 room made
physical (``default_home``; 蛋黃, without a nest, its ``no_nest`` variant: A2), with the
household (nest, a visible hand, the family's bodies, the world battery, the toys) and the
sensors (the camera behind ``media.frame``, the ToF behind ``tofd``).

* **The truth is in world coordinates** (the home's own; ``default_home``: the same as Duck
  Core's room). Each body frame carries ``truth`` — ``x_mm``, ``y_mm``, ``yaw_mrad`` of the
  physics body — and when a world operation last moved it (``placed_at``, ``placed_why``).
  Duck Core's plan is drawn moved by the layout's nest (or mat) pose (:class:`HomeFrame`;
  identity in ``default_home``). Nothing re-seats the world any more: phase 0's anchors are
  gone.
* **World operations move the body, and say so.** When the body starts it is put where Duck
  Core is (``x_place``: ``placed_why`` "start"); after time the physics did not live (a
  fast-forward) it is put where Duck Core's plan now is ("sync"), once the fast-forward stops.
  A spot taken by furniture moves to the nearest free one (:func:`spiral`).
* **A hand really carries the body** (D1). Duck Core's hand events — picked up, put on the
  floor, on the desk (= the mat, A3), into the nest, set upright — become scripts of the
  world's visible hand (``x_hand``, :func:`hand_command`): the shadow (the plan) jumps at once,
  the solid body arrives a few seconds later; held in the air it sits with its legs tucked.
  「扶起物理身體」 (:meth:`LiveBridge.upright`) is the everyday recovery after a physics fall
  (D2; 「重設物理身體」 stays as the fallback). A family member's touch comes from that
  person's body, which walks over first (:mod:`duckmind.simweb.family`, D3; the server runs
  it and tells :meth:`LiveBridge.after_call` what the hand already did).
* **The duck's-eye window** asks the world's camera for a frame (``media.frame``, through
  :class:`duckmind.eye.frames.EyeWindow`) only while the page's window is open (F4).
* **The curiosity demonstrations** (示範: bump, peck, kick; :mod:`duckmind.simweb.demo`) and the
  world's own buttons (roll a ball, tidy up) never involve Duck Core.
* **「用眼睛做」** (owner decision A6, phase 3; :mod:`duckmind.simweb.eyedemo`): the same curiosity
  step done by the duck's own processes — duck-eye and duck-body started against this body's
  sockets, one ``duck.curio`` step through the Act Runner — while this bridge's body sends robotd
  nothing (``LiveBody.passive``). The world only puts the thing roughly in front of the duck.

Duck Core believing it walked somewhere (into its nest) moves only the plan, never the body
(W05). On phase 0's body host (``DUCKMIND_LIVE_HOME=bodyhost``) there are no world ops: the
truth is then the flat floor's coordinates and nothing can move the body.

**The charging dock** (owner decision 2026-09-27) stands against a wall. In ``default_home`` Duck
Core's frame stays the room (so its imagined walls are the physical walls), and a nest duck still
believes its nest is at its origin (0, 0), the middle of the room, until phase 5 teaches it where
the dock is: its shadow goes to (0, 0) while the body sits in the dock by the wall, and the page
shows both. A duck that is in its nest when the body starts or after a fast-forward is put in the
physical dock (seated, facing the way the dock's style docks), not at the plan's (0, 0); a hand
putting it in its nest carries it into the dock.
"""

from __future__ import annotations

import heapq
import itertools
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from duckmind.duckcore.api import _motion
from duckmind.i18n import lst, msg
from duckmind.livebody import FRAME_JOINTS, LiveBody, LivePaths
from duckmind.simweb.eyecam import EyeCam, taipei_now

if TYPE_CHECKING:
    from duckmind.duckcore.api import AnySession

# Events after which nothing that was still waiting should play.
INTERRUPTS = frozenset(
    {
        "power.off",
        "sense.pickup",
        "sense.hold_head",
        "env.floor",
        "body.fell",
        "duck.nap_start",
        "pad.play_start",
    }
)
SYNC_EVERY_S = 5.0  # with nothing to play, re-ask for Duck Core's posture this often
DEMO_S = 7.0  # how long the physics demonstration walk takes (stand up, walk, settle)
# A fast-forward steps once a second; its placement ("sync") waits until the steps stop. The
# page asks on a 1 s timer and skips a tick while a step is still being answered (a 15-minute
# step can take most of a second), so two steps can be 2 s apart: the quiet window is longer.
SYNC_QUIET_S = 3.0
# Where a placement looks when Duck Core's spot is taken (furniture, a wall, a toy).
PLACE_STEP_M = 0.1
PLACE_REACH_M = 0.8
RESTORE_MOVE_MM = (
    150  # a reloaded home puts the duck back where it was if that spot (or one this near) is free
)
# The light clock of the world's camera follows Duck Core's clock (x_lights): re-set when it
# drifts this far, and at least this often.
CLOCK_SLACK_S = 20.0
CLOCK_EVERY_S = 60.0
EYE_PERIOD_S = 2.0  # the page asks every 2 s while its window is open
EYE_IDLE_S = 6.0  # no ask for this long: the window is closed, no more frames
FROZEN_S = 1.0  # physics time standing still this long: the picture is of a stopped world
DEMO_LIMIT_S = 45.0
# Prototype mode (--brain neuro): the demonstrations would fight the neural duck for the body.
NEURO_BUSY_ZH = msg("neuro.hook.demo_busy")
EYE_DEMO_LIMIT_S = 180.0  # 「用眼睛做」: start the processes, stand, look, one step (≤ 100 s)
TICK_S = 0.1  # a demonstration's pace
ALREADY_M = 0.06  # a put-down this close to where the body already is needs no hand
# eye_image's tag while the world's camera takes the first picture for a window just opened
# (the server answers 202: nothing is wrong, it comes in a moment).
EYE_PENDING = "pending"
# Prototype mode (fid:page 2026-09-28): the body server is read a little faster than its 50 Hz
# control tick (a read that lands between two steps is dropped as the same sample), so the page's
# 3D view gets (nearly) every physics sample (livebody FRAMES_*), not ~27 a second.
NEURO_SAMPLE_HZ = 60
HAND_TAKE = frozenset({"pickup", "place", "upright"})  # the hand moves the body


def _in_posture(policy: Any, posture: str | None) -> bool:
    """Whether the body's running policy is the posture asked for (None: any)."""
    if posture is None:
        return True
    return policy == "sit" if posture == "sit" else policy in ("stand", "walk")


def hand_command(event: Any, *, nest: bool) -> dict[str, Any] | None:
    """What the world's hand does for one Duck Core event (owner decisions D1, A2, A3), or None.

    ``nest``: the duck has a nest. Poses are Duck Core's (``pose``: [x m, y m, heading rad] in
    its own frame; the bridge turns them into the world's). A duck with a nest is held up in
    the air after a pickup until someone puts it down; one without (蛋黃) is lifted and set back
    on its desk, which is the mat on the floor (A3). Put on the floor: where Duck Core says.
    Into the nest by a hand (never walked in by itself: that is only what it believes, W05).
    Set upright where it lies."""
    kind = event.kind
    payload = event.payload
    if kind == "sense.pickup":
        return {"do": "pickup"} if nest else {"do": "place", "to": "desk"}
    if kind == "env.floor":
        if payload.get("why") == "pickup":
            return None if nest else {"do": "place", "to": "desk"}
        if payload.get("floor"):
            pose = payload.get("pose")
            if not pose:
                return {"do": "place", "to": "here"}
            return {"do": "place", "to": "floor", "pose": [float(value) for value in pose]}
        return {"do": "pickup"} if nest else {"do": "place", "to": "desk"}
    if kind == "nest.docked" and payload.get("by") not in ("self", "reseat"):
        return {"do": "place", "to": "nest"} if nest else None
    if kind == "sense.set_upright":
        return {"do": "upright"}
    return None


def done_already(command: dict[str, Any], done: dict[str, Any] | None) -> bool:
    """Whether a family visit's hand already did ``command`` (``Due.done_by_hand``): the body
    is not moved a second time for Duck Core's echo of it."""
    if not done:
        return False
    did = str(done.get("hand") or "")
    if did == "pickup":
        return command.get("do") == "pickup"
    if did == "place":
        return command.get("do") == "place" and command.get("to") == done.get("to")
    if did == "upright":
        return command.get("do") == "upright"
    if did == "pickup+place":
        return command.get("do") == "pickup" or command.get("to") == "here"
    return False


def _posture_of(session: AnySession) -> str | None:
    """The body posture Duck Core's state asks for (None: leave it, e.g. lying fallen)."""
    s = session.core.s
    if not s.awake or s.posture == "fallen":
        return None
    return "stand" if s.posture in {"stand", "walk"} else "sit"


def yaw_of(quat: Sequence[float]) -> float:
    """The heading (about the vertical) of a MuJoCo quaternion (w, x, y, z), radians: where
    the nose points on the floor (the page's ``yawOf`` is the same formula). It flips by π once
    the nose tips past straight down or up (see heading_of)."""
    w, x, y, z = (float(value) for value in quat[:4])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading_of(quat: Sequence[float]) -> float:
    """The turn about the vertical in an orientation (its twist about world z), radians.

    Equal to :func:`yaw_of` for an upright body (to a fraction of a degree when it leans) and,
    unlike it, the same however far the trunk tips: a duck lying on its face or back still has
    the heading it fell with."""
    w, x, y, z = (float(value) for value in quat[:4])
    if abs(w) < 1e-9 and abs(z) < 1e-9:  # turned exactly upside down: no twist to read
        return yaw_of(quat)
    return _wrap(2 * math.atan2(z, w))


def physics_pose(frame: dict[str, Any] | None) -> tuple[float, float, float] | None:
    """Where a body frame has the trunk in the physics world: x, y (m) and heading (rad)."""
    if not frame or frame.get("trunk") is None or frame.get("quat") is None:
        return None
    trunk = frame["trunk"]
    return float(trunk[0]), float(trunk[1]), yaw_of(frame["quat"])


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def spiral(
    x: float, y: float, step: float = PLACE_STEP_M, reach: float = PLACE_REACH_M
) -> Iterator[tuple[float, float]]:
    """Spots to try for a placement: (x, y) first, then rings ``step`` apart out to ``reach``,
    about ``step`` apart along each ring (nearest first)."""
    yield x, y
    rings = int(round(reach / step))
    for ring in range(1, rings + 1):
        radius = ring * step
        count = max(8, int(round(2 * math.pi * radius / step)))
        for k in range(count):
            angle = 2 * math.pi * k / count
            yield x + radius * math.cos(angle), y + radius * math.sin(angle)


@dataclass(frozen=True)
class HomeFrame:
    """Where Duck Core's own origin lies in the world (the layout's nest, else its mat;
    ``default_home``: identity). Metres and radians."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @classmethod
    def of(cls, layout: Any) -> HomeFrame:
        frame = getattr(layout, "frame", None)
        if frame is None:
            return cls()
        return cls(frame.x_mm / 1000.0, frame.y_mm / 1000.0, frame.heading_mrad / 1000.0)

    def to_world(self, pose: Sequence[float | None]) -> tuple[float, float, float]:
        """A pose in Duck Core's frame ([x, y(, heading)]) in the world."""
        px, py = float(pose[0] or 0.0), float(pose[1] or 0.0)
        heading = float(pose[2] or 0.0) if len(pose) > 2 and pose[2] is not None else 0.0
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * px - s * py, self.y + s * px + c * py, _wrap(heading + self.yaw)

    def to_plan(self, pose: Sequence[float]) -> tuple[float, float, float]:
        """A world pose in Duck Core's frame."""
        dx, dy = float(pose[0]) - self.x, float(pose[1]) - self.y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        heading = float(pose[2]) if len(pose) > 2 else 0.0
        return c * dx + s * dy, -s * dx + c * dy, _wrap(heading - self.yaw)

    def view(self) -> dict[str, int]:
        return {
            "x_mm": round(self.x * 1000),
            "y_mm": round(self.y * 1000),
            "heading_mrad": round(self.yaw * 1000),
        }


def load_layout(home: str | None, mjcf_dir: str | Path | None = None) -> Any:
    """The home's layout (for its frame and the page), or None (the flat floor, a body host,
    a home that cannot be read here). ``mjcf_dir``: the official MJCF folder of the live body's
    checkout, for homes built on an official scene (the apartment) when
    DUCKMIND_MICRODUCK_MJCF is not set in this process."""
    if not home or home == "flat":
        return None
    from duckmind.world import homelayout

    for folder in (None, mjcf_dir):
        try:
            return homelayout.load(home, mjcf_dir=None if folder is None else str(folder))
        except Exception:
            continue
    return None


def official_mjcf(paths: LivePaths) -> Path:
    """The official microduck MJCF folder inside the live body's microduck_rl checkout."""
    return Path(paths.microduck_rl) / "src" / "mjlab_microduck" / "robot" / "microduck"


def plain_error(error: str | None) -> str | None:
    """What went wrong with the physics body, in plain words for the owner (``error`` keeps the
    program's own text for whoever debugs it)."""
    if not error:
        return None
    text = str(error)
    if "拒絕啟動" in text:
        return msg("web.live.plain_error.23fa82")
    if "live body needs" in text:
        return msg("web.live.plain_error.777d08")
    if "robotd" in text:
        return msg("web.live.plain_error.23f9f7")
    if "body oracle" in text or "duck-body" in text:
        return msg("web.live.plain_error.941e54")
    if "never reached" in text:
        return msg("web.live.plain_error.656743")
    return msg("web.live.plain_error.6a244e")


def _world_words(error: BaseException) -> str:
    """A world op's refusal in the world's own words (its ValueErrors are plain Chinese)."""
    text = str(error)
    if "'error':" in text:
        text = text.split("'error':", 1)[1].strip(" }'\"")
    if isinstance(error, ConnectionError) or "Connection" in type(error).__name__:
        return msg("web.live.world_words")
    return text or type(error).__name__


class WorldPort:
    """``call(op, **params)`` on the running body's world (for family.py and demo.py)."""

    def __init__(self, bridge: LiveBridge) -> None:
        self.bridge = bridge

    def call(self, op: str, **params: Any) -> dict[str, Any]:
        body = self.bridge.body
        call = getattr(body, "world_call", None)
        if call is None or not self.bridge.ready:
            raise ConnectionError(msg("web.live.WorldPort.call"))
        return dict(call(op, **params))


class WindowEye:
    """The duck's-eye window as a ``media.frame`` client (owner decision F4): a frame is
    fetched from the world's camera only while the page keeps asking (:meth:`touch`), then
    made into the page's JPEG. The same ``latest()`` / ``status()`` as the phase-0
    :class:`~duckmind.simweb.eyecam.EyeCam`, plus ``window: True`` (frames on demand)."""

    def __init__(
        self,
        socket_path: str,
        *,
        room_light: Callable[[], dict[str, Any]],
        frozen: Callable[[], bool],
    ) -> None:
        from duckmind.eye.frames import EyeWindow

        self._room_light = room_light
        self._frozen = frozen
        self._lock = threading.Lock()
        self._jpeg: dict[str, Any] | None = None
        self.window = EyeWindow(
            socket_path,
            period_s=EYE_PERIOD_S,
            idle_s=EYE_IDLE_S,
            step=2,
            on_frame=self._encode,
        )

    def _encode(self, result: dict[str, Any]) -> None:
        from duckmind.simweb import jpeg

        rgb = result.get("rgb")
        if rgb is None:
            return
        data, how = jpeg.encode(rgb, 80)
        header = result.get("header") or {}
        wall = float(header.get("captured_at_unix_us") or 0) / 1e6 or time.time()
        with self._lock:
            count = (self._jpeg or {}).get("frames", 0) + 1
            self._jpeg = {
                "jpeg": data,
                "encoder": how,
                "wall": wall,
                "frames": count,
                "t": None,
                "light": result.get("light"),
                "render_ms": result.get("latency_ms"),
            }

    def start(self) -> None:
        self.window.start()

    def stop(self) -> None:
        self.window.stop()

    def touch(self) -> None:
        self.window.touch()

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._jpeg) if self._jpeg is not None else None

    def status(self) -> dict[str, Any]:
        state = self.window.status()
        latest = self.latest() or {}
        room = self._room_light() or {}
        frozen = self._frozen()
        note = "" if latest else str(state.get("note") or "")
        if not state.get("open") and latest:
            note = msg("web.live.WindowEye.status.81de2f")
        return {
            "running": True,
            "window": True,
            "t": None,
            "wall": latest.get("wall"),
            "light": latest.get("light") or {},
            "label": room.get("label"),
            "lux": room.get("lux"),
            "render_ms": latest.get("render_ms"),
            "frames": int(latest.get("frames") or 0),
            "frozen": frozen,
            "error": None,
            "error_detail": state.get("error_detail") if state.get("error") else None,
            "note": (msg("web.live.WindowEye.status.b1db29") if frozen else note)
            or state.get("error")
            or "",
        }


class NoEye:
    """The world has a camera (``x_sensors``) that did not start: the window says so, instead of
    another renderer drawing a scene that is not the home the physics runs."""

    window = False

    def __init__(self, why: str, detail: str = "") -> None:
        self.why = why
        self.detail = detail

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def latest(self) -> dict[str, Any] | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "running": False,
            "window": False,
            "t": None,
            "wall": None,
            "light": {},
            "label": None,
            "lux": None,
            "render_ms": None,
            "frames": 0,
            "frozen": False,
            "error": self.why,
            "error_detail": self.detail or None,
            "note": "",
        }


class LiveBridge:
    def __init__(
        self,
        paths: LivePaths,
        side: int = 1,
        *,
        eye: bool = True,
        home: str | None = None,
        nest: bool | None = None,
        tof: bool | None = None,
    ) -> None:
        self.paths = paths
        self.side = side
        # The home asked for (None: DUCKMIND_LIVE_HOME, else default_home), whether the duck
        # has a nest (its home's variant) and a ToF module (the sensors'). Set before start().
        self.home_asked = home
        self.nest = nest
        self.tof = tof
        self.body = LiveBody(paths, side=side, seated=True, home=home, nest=nest, tof=tof)
        self.error: str | None = None
        self.ready = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._start, args=(self.body,), daemon=True)
        # Motions and world operations waiting for their moment: (monotonic due, tie, item).
        self._due: list[tuple[float, int, dict[str, Any]]] = []
        self._order = itertools.count()
        self._wake = threading.Condition()
        self._closing = False
        self._scheduler = threading.Thread(target=self._play_due, daemon=True)
        self._synced_at = 0.0
        self.demo_until = 0.0  # epoch seconds: a physics demonstration has the body
        # A body that just became ready has not been told Duck Core's state yet (whatever
        # happened while it started was dropped): the server resyncs it (take_fresh).
        self._fresh = False
        # The duck's-eye camera of the running body (None while the body starts).
        self.eye_enabled = eye
        self.eye: Any = None
        # Duck Core's local time, for the light the eye sees (the server sets it).
        self.clock: Callable[[], datetime] | None = None
        # The home's frame (Duck Core's origin in the world) and layout, for the body's home.
        self._layout: Any = None
        self._layout_home: str | None = "?"
        # The last world operation that moved the body (x_place: start / sync) and what the
        # hand was last asked to do, for the page.
        self._placed: dict[str, Any] | None = None
        self.hand_note: dict[str, Any] | None = None
        self.world_note: dict[str, Any] | None = None
        # A fast-forward's placement, waiting for the steps to stop (see skipped).
        self.sync_quiet_s = SYNC_QUIET_S
        self._sync_lock = threading.Lock()
        self._sync_pending: tuple[tuple[float, float, float], str | None, bool, float] | None = None
        self._sync_thread: threading.Thread | None = None
        # The posture Duck Core last wanted (the end of a demonstration returns to it).
        self._want: str | None = None
        # The curiosity demonstration running (示範 or 「用眼睛做」) and its last result.
        self._demo: Any = None
        # Where the duck's learned floor is kept between 「用眼睛做」 runs (the server sets it next
        # to the duck's saved state; None: the body's own run folder, gone with it).
        self.eye_dir: Path | None = None
        self.demo_result: dict[str, Any] | None = None
        # Prototype mode (``--brain neuro``): the neural duck's driver
        # (:class:`duckmind.simweb.neuro_live.NeuroDriver`) drives the body instead of Duck
        # Core's motions; only the world's hand and light clock still come from here. None:
        # the normal mode, unchanged.
        self.neuro: Any = None
        # The light clock last sent to the world (duck local, wall, rate).
        self._clock_sent: tuple[datetime, float, float] | None = None
        # Physics time of the latest frame and when it last moved on (monotonic).
        self._frame_t: Any = None
        self._frame_t_at = 0.0
        # Old bodies being stopped after a restart (stop() waits for them).
        self._retiring: list[threading.Thread] = []
        # Where the duck was when its physics reloaded for a changed home (layout editor): the new body
        # starts there if that spot is free in the new home (x m, y m, yaw rad), else at its spawn.
        self.restore_pose: tuple[float, float, float] | None = None
        self.restored: str | None = None  # "there" | "spawn" after such a reload (for the page)
        self.restoring = False  # the duck is still being put back (the page waits for it)

    # ── lifecycle ───────────────────────────────────────────────────────

    def configure(self, *, nest: bool | None = None, tof: bool | None = None) -> None:
        """Before :meth:`start`: the duck this body is for (its nest decides its home's
        variant, its head whether the ToF module is fitted)."""
        if self._thread.is_alive():
            raise RuntimeError("configure() the bridge before start()")
        self.nest = nest
        self.tof = tof
        self.body = LiveBody(
            self.paths, side=self.side, seated=True, home=self.home_asked, nest=nest, tof=tof
        )
        self._thread = threading.Thread(target=self._start, args=(self.body,), daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._scheduler.start()

    def _start(self, body: LiveBody) -> None:
        try:
            body.start()
            if body is self.body:
                self.ready = True
                self._fresh = True
                self._start_eye(body)
                if self.neuro is not None:
                    self._neuro_start(body)
        except Exception as error:  # the page falls back to the idealized view
            if body is self.body:
                self.error = f"{type(error).__name__}: {error}"

    def _neuro_start(self, body: LiveBody) -> None:
        """Prototype mode: the neural duck starts seated on the family's mat in the open room
        (a start placement, the world op the normal mode's "start" uses too; the spawn is the
        charging dock, nose to its backboard, which a duck leaves only by reversing out), then
        its driver takes the body. ``neuro.start_at`` "nest" keeps the spawn. The body server
        is read at every control tick then (NEURO_SAMPLE_HZ): the page draws each sample."""
        if hasattr(body, "sample_hz"):
            body.sample_hz = float(NEURO_SAMPLE_HZ)
        want, self.restore_pose = self.restore_pose, None
        if want is not None and getattr(body, "world_call", None) is not None:
            # the home changed under the duck: back where it was when that spot is still free
            try:
                note = self.place(want, "start")
                moved = note.get("moved_mm")
                if note.get("ok") and moved is not None and moved <= RESTORE_MOVE_MM:
                    self.restored = "there"
                    self.neuro.attach(body)
                    self.restoring = False
                    return
            except Exception:
                pass
            self.restored = "spawn"  # (taken: the spawn below)
        try:
            self._neuro_spawn(body)
        finally:
            self.restoring = False

    def _neuro_spawn(self, body: LiveBody) -> None:
        if getattr(self.neuro, "start_at", "mat") != "nest":
            mat = getattr(self.layout, "mat", None)
            if mat is not None and getattr(body, "world_call", None) is not None:
                try:
                    self.place((mat.x_mm / 1000.0, mat.y_mm / 1000.0, 0.0), "start")
                except Exception:
                    pass  # (it starts at the spawn instead; the page shows where)
        self.neuro.attach(body)  # the neural duck takes the body

    @property
    def home(self) -> str | None:
        """The home the body's world runs (None: phase 0's body host)."""
        return getattr(self.body, "home", None)

    @property
    def layout(self) -> Any:
        home = self.home
        if home != self._layout_home:
            self._layout = load_layout(home, official_mjcf(self.paths))
            self._layout_home = home
        return self._layout

    @property
    def frame(self) -> HomeFrame:
        """Duck Core's origin in the world (identity on a flat floor)."""
        return HomeFrame.of(self.layout)

    def world_port(self) -> WorldPort:
        return WorldPort(self)

    def world_ready(self) -> bool:
        """The body runs in a world with the household (a hand and people to ask)."""
        if not self.ready or self.error is not None or getattr(self.body, "error", None):
            return False
        ops = (getattr(self.body, "host_info", None) or {}).get("ops") or []
        return "x_hand" in ops and "x_people" in ops

    # ── the duck's-eye camera ──────────────────────────────────────────────

    def _eye_clock(self) -> datetime:
        clock = self.clock
        return clock() if clock is not None else taipei_now()

    def _media_socket(self, body: Any) -> tuple[bool, str | None, str]:
        """Whether the world has a camera (x_sensors), its media.frame socket (None when it did
        not start) and, then, why not (plain words)."""
        ops = (getattr(body, "host_info", None) or {}).get("ops") or []
        call = getattr(body, "world_call", None)
        if "x_sensors" not in ops or call is None:
            return False, None, ""
        try:
            sensors = call("x_sensors")
        except Exception as error:
            return (
                True,
                None,
                msg("web.live.LiveBridge.media_socket.37ea19", error=_world_words(error)),
            )
        media = (sensors.get("media") or {}).get("socket")
        if media:
            return True, str(media), ""
        said = [str(text) for text in sensors.get("errors") or []]
        eye = sensors.get("eye") or {}
        if eye.get("error"):
            said.append(str(eye.get("error")))
        return True, None, "；".join(said) or msg("web.live.LiveBridge.media_socket.cf3cf5")

    def _start_eye(self, body: LiveBody) -> None:
        """The head camera of this body: the world's own camera through media.frame, asked for
        frames only while the page's window is open (F4); on phase 0's body host (no world
        camera), phase 0's own render of the scene the physics loaded. A world whose camera did
        not start gets no picture and the reason — never another scene's render."""
        if not self.eye_enabled or not hasattr(body, "frame"):
            return
        has_camera, socket, why = self._media_socket(body)
        eye: Any
        if socket is not None:
            eye = WindowEye(socket, room_light=self._room_light, frozen=self._frozen)
        elif has_camera:
            eye = NoEye(msg("web.live.LiveBridge.start_eye"), why)
        else:
            scene = (getattr(body, "host_info", None) or {}).get("scene")
            eye = EyeCam(body.frame, scene=scene, clock=self._eye_clock)
        with self._lock:
            old, self.eye = self.eye, eye
        if old is not None:
            threading.Thread(target=old.stop, daemon=True).start()
        eye.start()

    def _stop_eye(self, wait: bool = False) -> None:
        with self._lock:
            eye, self.eye = self.eye, None
        if eye is None:
            return
        if wait:
            eye.stop()
        else:  # (never inside a request: joining the render thread takes up to a frame)
            threading.Thread(target=eye.stop, daemon=True).start()

    def world_truth(self) -> dict[str, Any] | None:
        """The world's latest truth (x_world), whole: None before its first answer, without a
        world, or when it is stale. What the household reads of the pack and the hand."""
        return self._world()

    def _world(self) -> dict[str, Any] | None:
        world = getattr(self.body, "world", None)
        return world() if callable(world) else None

    def _room_light(self) -> dict[str, Any]:
        """The light of the room the duck is in (the world's lights section: label, lux)."""
        lights = (self._world() or {}).get("lights") or {}
        rooms = lights.get("rooms") or {}
        if not rooms:
            return {}
        truth = self.truth(self._latest_frame())
        layout = self.layout
        room = None
        if truth is not None and layout is not None and hasattr(layout, "room_at"):
            room = layout.room_at(truth["x_mm"], truth["y_mm"])
        chosen = rooms.get(room) if room else None
        return dict(chosen or next(iter(rooms.values())))

    def _frozen(self) -> bool:
        """Whether the physics time has stood still for FROZEN_S (the world is stopped)."""
        frame = self._latest_frame()
        t = (frame or {}).get("t")
        now = time.monotonic()
        if t is None or t != self._frame_t:
            self._frame_t, self._frame_t_at = t, now
            return False
        return now - self._frame_t_at >= FROZEN_S

    def eye_status(self) -> dict[str, Any]:
        """The duck's-eye picture for the page, without the picture: when it was taken (``t``
        the physics time where known, ``age_s`` seconds ago), the light level its meter read
        (``level``, ``level_zh``; ``adapting`` while the exposure catches up), the room's light
        (``label``, ``lux``), the fetch or render cost, ``frozen`` while the physics stands
        still, ``window`` when frames come only while the page's window is open (F4) and, when
        there is no picture to be had, why (``error`` in plain words, ``error_detail`` the
        program's own; ``note`` while waiting)."""
        eye = self.eye
        out: dict[str, Any] = {
            "t": None,
            "age_s": None,
            "level": None,
            "level_zh": None,
            "adapting": False,
            "label": None,
            "lux": None,
            "render_ms": None,
            "frames": 0,
            "running": False,
            "window": False,
            "frozen": False,
            "error": None,
            "error_detail": None,
            "note": None,
        }
        if eye is None:
            if not self.eye_enabled:
                out["error"] = msg("web.live.LiveBridge.eye_status.7cfbe7")
            elif self.error is not None:
                out["error"] = msg("web.live.LiveBridge.eye_status.f1967f")
            else:
                out["note"] = msg("web.live.LiveBridge.eye_status.2402bc")
            return out
        status = eye.status()
        light = status.get("light") or {}
        wall = status.get("wall")
        out.update(
            t=status.get("t"),
            age_s=round(max(0.0, time.time() - float(wall)), 1) if wall else None,
            level=light.get("level"),
            level_zh=light.get("level_zh"),
            adapting=bool(light.get("adapting")),
            label=status.get("label"),
            lux=status.get("lux"),
            render_ms=status.get("render_ms"),
            frames=int(status.get("frames") or 0),
            running=bool(status.get("running")),
            window=bool(status.get("window")),
            frozen=bool(status.get("frozen")),
            error=status.get("error"),
            error_detail=status.get("error_detail") if status.get("error") else None,
            note=status.get("note") or None,
        )
        if self.error is not None or self.body.error is not None:
            # (the camera shows the last pose the body had: say so instead of a fresh age)
            out["error"] = msg("web.live.LiveBridge.eye_status.e2d50a")
        return out

    def eye_image(self) -> tuple[bytes | None, str, str]:
        """The latest duck's-eye JPEG and a tag for it, or (None, "", why there is none), or
        (None, EYE_PENDING, words) while the world's camera takes the first picture. Asking is
        what keeps the world's camera window open (F4)."""
        eye = self.eye
        if eye is None:
            status = self.eye_status()
            return None, "", str(status["error"] or status["note"])
        touch = getattr(eye, "touch", None)
        if callable(touch):
            touch()
        latest = eye.latest()
        if latest is None:
            status = eye.status()
            broken = self.error is not None or getattr(self.body, "error", None) is not None
            if callable(touch) and not status.get("error") and not broken:
                # (the window just opened: the camera renders the first picture now)
                return None, EYE_PENDING, msg("web.live.LiveBridge.eye_image.f69436")
            why = status.get("error") or msg(
                "web.live.LiveBridge.eye_image.50bbbc",
                status=status.get("note") or msg("web.live.LiveBridge.eye_image.69683a"),
            )
            return None, "", why
        tag = f'"{int(float(latest.get("wall") or 0) * 1000)}-{int(latest.get("frames") or 0)}"'
        return bytes(latest["jpeg"]), tag, ""

    # ── the truth: the physics body in the world ──────────────────────────

    def _latest_frame(self) -> dict[str, Any] | None:
        """The running body's latest frame (None while it starts, or for a body without any)."""
        frame = getattr(self.body, "frame", None) if self.ready else None
        return frame() if callable(frame) else None

    def truth(self, frame: dict[str, Any] | None) -> dict[str, Any] | None:
        """Where this body frame has the duck in the world (None without a frame): ``x_mm``,
        ``y_mm``, ``yaw_mrad`` (the home's world coordinates; ``default_home``: Duck Core's own
        room), ``placed_at`` (epoch ms of the last world operation that moved the body: None
        before the first) and ``placed_why`` ("start" | "sync"). Ground truth for the page and
        evaluators only, never for Duck Core: it reads the frame, nothing of the duck's own
        state and nothing of its plan."""
        phys = physics_pose(frame)
        if phys is None:
            return None
        placed = self._placed or {}
        return {
            "x_mm": round(phys[0] * 1000),
            "y_mm": round(phys[1] * 1000),
            "yaw_mrad": round(phys[2] * 1000),
            "frame": "world",
            "placed_at": placed.get("at_ms"),
            "placed_why": placed.get("why"),
        }

    def truth_now(self) -> dict[str, Any] | None:
        """:meth:`truth` of the latest frame (None while the body starts)."""
        return self.truth(self._latest_frame())

    def take_fresh(self) -> bool:
        """Whether the body became ready since the last call (it then needs Duck Core's
        power and posture, and to be put where the duck is: a restarted body sits at its
        spawn, whatever the duck is doing)."""
        fresh, self._fresh = self._fresh, False
        return fresh

    def restart(
        self,
        side: int | None = None,
        *,
        home: str | None = None,
        nest: bool | None = None,
        tof: bool | None = None,
        restore: tuple[float, float, float] | None = None,
    ) -> None:
        """A fresh physics body at its home's spawn (after a fall, or a new duck: ``nest`` and
        ``tof`` change with the duck, and so does its home's variant). ``restore``: where the duck
        was (world x m, y m, yaw rad) when the home itself changed (the layout editor): the new body
        starts there if that spot is free, else at the spawn."""
        if self.neuro is not None:
            self.neuro.detach()  # (it takes the new body once that one is ready)
        self._stop_eye()
        self._stop_demo()
        self.restore_pose = restore
        self.restored = None
        self.restoring = restore is not None
        with self._lock:
            old = self.body
            old_thread = self._thread
            self.side = self.side if side is None else side
            self.home_asked = self.home_asked if home is None else home
            self.nest = self.nest if nest is None else nest
            self.tof = self.tof if tof is None else tof
            self.ready = False
            self.error = None
            self.body = LiveBody(
                self.paths,
                side=self.side,
                seated=True,
                home=self.home_asked,
                nest=self.nest,
                tof=self.tof,
            )
            self._thread = threading.Thread(target=self._start, args=(self.body,), daemon=True)
        self._placed = None
        self.hand_note = None
        self._clock_sent = None
        with self._sync_lock:
            self._sync_pending = None
        self.demo_until = 0.0
        with self._wake:
            self._due.clear()

        def retire(body: LiveBody = old, starting: threading.Thread = old_thread) -> None:
            # (a body still starting is stopped once its start has finished or failed: stopped
            # halfway, its processes and its /private/tmp/dm-live-* folder could outlive it)
            if starting.is_alive():
                starting.join(timeout=90)
            body.stop()

        retiring = threading.Thread(target=retire, daemon=True)
        self._retiring.append(retiring)
        retiring.start()
        self._thread.start()

    # ── world operations: placing the body, the hand, the light clock ───────

    def _call(self, op: str, **params: Any) -> dict[str, Any]:
        call = getattr(self.body, "world_call", None)
        if call is None:
            raise ConnectionError(msg("web.live.LiveBridge.call"))
        return dict(call(op, **params))

    def place(self, pose: Sequence[float], why: str) -> dict[str, Any]:
        """Put the physics body at ``pose`` (world metres and radians) at rest — a world
        operation (``x_place``; the hand lets go first). A spot that is taken moves to the
        nearest free one within PLACE_REACH_M. Records and returns what happened."""
        at_ms = int(time.time() * 1000)
        x, y = float(pose[0]), float(pose[1])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        note: dict[str, Any] = {
            "at_ms": at_ms,
            "why": why,
            "ok": False,
            "asked_mm": [round(x * 1000), round(y * 1000)],
            "x_mm": None,
            "y_mm": None,
            "moved_mm": None,
            "blocked_by": [],
            "why_zh": [],
        }
        if not getattr(self.body, "home", None) or getattr(self.body, "world_call", None) is None:
            note["why_zh"] = [msg("web.live.LiveBridge.place.4af93e")]
            self._placed = note
            return note
        try:
            hand = (self._world() or {}).get("hand") or {}
            if hand.get("script") is not None or hand.get("holding"):
                self._call("x_hand", do="cancel")
        except Exception:
            pass  # (the hand lets go by itself before the next step anyway)
        policy = getattr(self.body, "policy", "sit")
        keyframe = "STAND" if policy in ("stand", "walk") else "SIT"
        blocked: list[str] = []
        refused: list[str] = []
        asked = [round(x * 1000), round(y * 1000)]
        for cx, cy in spiral(x, y):
            try:
                answer = self._call(
                    "x_place",
                    x_mm=round(cx * 1000),
                    y_mm=round(cy * 1000),
                    yaw_mrad=round(yaw * 1000),
                    keyframe=keyframe,
                    # (a stand-in spot must be indoors, off a drop and not behind a wall from
                    # the asked one: the world refuses the others before any physics)
                    **({"from_mm": asked} if (cx, cy) != (x, y) else {}),
                )
            except Exception as error:
                note["why_zh"] = [
                    msg("web.live.LiveBridge.place.f0f3a5", error=_world_words(error))
                ]
                break
            if answer.get("ok"):
                moved = math.hypot(cx - x, cy - y)
                names = self._plain_names(blocked)
                first: str
                if names:
                    first = msg("web.live.LiveBridge.place.2d37fc", names=lst(names[:3]))
                else:
                    first = refused[0] if refused else msg("web.live.LiveBridge.place.cae680")
                note.update(
                    ok=True,
                    x_mm=int(answer.get("x_mm", round(cx * 1000))),
                    y_mm=int(answer.get("y_mm", round(cy * 1000))),
                    moved_mm=round(moved * 1000),
                    blocked_by=blocked,
                    why_zh=[msg("web.live.LiveBridge.place.a81d93", first=first, moved=moved * 100)]
                    if moved > 0.005
                    else [],
                )
                break
            for name in answer.get("blocked_by") or []:
                if name not in blocked:
                    blocked.append(str(name))
            if answer.get("refused") and cx == x and cy == y:
                refused = [str(text) for text in answer.get("why_zh") or []]
        else:
            names = self._plain_names(blocked)
            reach = msg("web.live.LiveBridge.place.107d4a", PLACE_REACH_M=PLACE_REACH_M * 100)
            if names:
                said = msg("web.live.LiveBridge.place.0da1b5", reach=reach, names=lst(names[:3]))
            elif refused:
                said = msg("web.live.LiveBridge.place.2441fe", reach=reach, refused=refused[0])
            else:
                said = msg("web.live.LiveBridge.place.46c584", reach=reach)
            note["why_zh"] = [said]
        seated = self._seat_note(x, y)
        if seated and note["ok"]:
            note["why_zh"] = [seated, *note["why_zh"]]
            note["seat"] = True
        self._placed = note
        return note

    def _seat_note(self, x: float, y: float) -> str | None:
        """Plain words for a placement into the charging place (the body goes to the physical
        dock because the duck sits in its nest): and, where Duck Core's frame does not put its
        nest there (default_home), that it believes its nest is elsewhere."""
        layout = self.layout
        nest = getattr(layout, "nest", None)
        if nest is None:
            return None
        seat = nest.seat
        if math.hypot(x - seat.x, y - seat.y) > 0.002:
            return None
        if not getattr(nest, "is_dock", False):
            return msg("web.live.LiveBridge.seat_note.65caa4")
        home = self.frame.to_world((0.0, 0.0))
        if math.hypot(home[0] - seat.x, home[1] - seat.y) > 0.3:
            return msg("web.live.LiveBridge.seat_note.102868")
        return msg("web.live.LiveBridge.seat_note.66a2e9")

    def _plain_names(self, names: Sequence[str]) -> list[str]:
        """The world's geom names as the owner would say them (沙發、牆、玩具…), once each."""
        layout = self.layout
        zh_of = getattr(layout, "zh_of", None)
        out: list[str] = []
        for name in names:
            said = zh_of(name) if callable(zh_of) else None
            said = said or msg("web.live.LiveBridge.plain_names")
            if said not in out:
                out.append(said)
        return out

    def _hand(self, command: dict[str, Any]) -> None:
        """One script of the world's hand (Duck Core's poses turned into the world's)."""
        request = dict(command)
        pose = request.pop("pose", None)
        if pose is not None:
            wx, wy, wyaw = self.frame.to_world(pose)
            request.update(
                x_mm=round(wx * 1000), y_mm=round(wy * 1000), yaw_mrad=round(wyaw * 1000)
            )
        note: dict[str, Any] = {
            "at_ms": int(time.time() * 1000),
            "do": request.get("do"),
            "to": request.get("to"),
        }
        if self._already_there(request):
            # (蛋黃's desk is the mat on the floor: "put on the floor" from the desk moves
            # nothing, and no hand comes for it)
            self.hand_note = {**note, "ok": True, "skipped": True, "why_zh": []}
            return
        if request.get("do") == "upright" and self._upright_now():
            # Duck Core's fall can be one only its plan had (a dice roll): the physics body
            # sits or stands fine, and no hand moves it for a fall that did not happen.
            self.hand_note = {
                **note,
                "ok": True,
                "skipped": True,
                "why_zh": [msg("web.live.LiveBridge.hand.1f729e")],
            }
            return
        try:
            answer = self._call("x_hand", **request)
            # (the hand's own bookkeeping — "another script replaced mine" — is not news)
            said = [
                text
                for text in answer.get("why_zh") or []
                if not str(text).startswith("換成新的動作")
            ]
            note.update(ok=bool(answer.get("ok", True)), eta_s=answer.get("eta_s"), why_zh=said)
        except Exception as error:
            note.update(
                ok=False, why_zh=[msg("web.live.LiveBridge.hand.b44808", error=_world_words(error))]
            )
        self.hand_note = note

    def _already_there(self, request: dict[str, Any]) -> bool:
        """A put-down whose spot the body already sits or stands on, upright and not held:
        within ALREADY_M of it (the mat for the desk, the nest's middle, a floor spot)."""
        if request.get("do") != "place":
            return False
        frame = self._latest_frame()
        if not frame or frame.get("trunk") is None or frame.get("quat") is None:
            return False
        world = self._world() or {}
        if (world.get("hand") or {}).get("holding") or getattr(self.body, "held", False):
            return False
        w, x, y, z = (float(value) for value in frame["quat"][:4])
        if 1 - 2 * (x * x + y * y) < math.cos(math.radians(25)):  # tipped over: the hand rights it
            return False
        to = request.get("to")
        layout = self.layout
        if to in ("desk", "mat"):
            spot = getattr(layout, "mat", None)
            target = (spot.x_mm / 1000, spot.y_mm / 1000) if spot is not None else (0.0, 0.0)
        elif to == "nest":
            spot = getattr(layout, "nest", None)
            if spot is None:
                return False
            seat = spot.seat  # where a docked duck sits (the open nest: its pad's middle)
            target = (seat.x, seat.y)
        elif to == "floor" and "x_mm" in request:
            target = (request["x_mm"] / 1000, request["y_mm"] / 1000)
        else:
            return False
        trunk = frame["trunk"]
        return math.hypot(float(trunk[0]) - target[0], float(trunk[1]) - target[1]) < ALREADY_M

    def _upright_now(self) -> bool:
        """The physics body is upright (trunk tilt under 25°) and robotd does not say it fell."""
        frame = self._latest_frame()
        if not frame or frame.get("quat") is None:
            return False
        _, x, y, _ = (float(value) for value in frame["quat"][:4])
        if 1 - 2 * (x * x + y * y) < math.cos(math.radians(25)):
            return False
        try:
            fallen = bool((self.body.status() or {}).get("fallen"))
        except Exception:
            fallen = False
        return not fallen

    def upright(self) -> dict[str, Any]:
        """「扶起物理身體」 (owner decision D2): the world's hand takes the fallen physics body
        where it lies, turns it upright at the heading it fell with and sets it down. A world
        operation: Duck Core is not told."""
        if not self.ready:
            raise ValueError(msg("web.live.LiveBridge.upright.d69176"))
        try:
            answer = self._call("x_hand", do="upright")
        except Exception as error:
            raise ValueError(
                msg("web.live.LiveBridge.upright.87cae0", error=_world_words(error))
            ) from error
        self.hand_note = {
            "at_ms": int(time.time() * 1000),
            "do": "upright",
            "to": "upright",
            "ok": True,
            "eta_s": answer.get("eta_s"),
            "why_zh": list(answer.get("why_zh") or []),
        }
        return answer

    def place_prop(self, thing: str, x_mm: int, y_mm: int, on: str = "floor") -> dict[str, Any]:
        """Put a household thing at a caller-chosen spot (P1d-4 calibration only).

        Thin wrapper over the world's own ``x_place_prop`` (the same op
        ``demo.py`` uses for truth positioning). Places the thing and nothing
        else: no drive/skill command, no robotd write, no safety-chain change.
        The brain is not told; it may notice the thing only through its senses."""
        if not self.ready:
            raise ValueError(msg("web.live.LiveBridge.world_action"))
        try:
            answer = self._call(
                "x_place_prop",
                id=str(thing),
                x_mm=int(x_mm),
                y_mm=int(y_mm),
                on=str(on),
            )
        except Exception as error:
            raise ValueError(_world_words(error)) from error
        self.world_note = {"at_ms": int(time.time() * 1000), "op": "x_place_prop", "answer": answer}
        return answer

    def world_action(self, op: str, **params: Any) -> dict[str, Any]:
        """A world button that is not Duck Core's (roll a ball, tidy up): the world's answer,
        or ValueError in the world's words."""
        if not self.ready:
            raise ValueError(msg("web.live.LiveBridge.world_action"))
        try:
            answer = self._call(op, **params)
        except Exception as error:
            raise ValueError(_world_words(error)) from error
        self.world_note = {"at_ms": int(time.time() * 1000), "op": op, "answer": answer}
        return answer

    def set_clock(self, local: datetime, rate: float) -> None:
        """The world camera's light follows Duck Core's clock (``rate``: its seconds per wall
        second; 0 while paused). Sent when it drifts or the rate changes (asynchronously)."""
        if not self.ready or getattr(self.body, "world_call", None) is None:
            return
        now = time.time()
        sent = self._clock_sent
        if sent is not None:
            expected = sent[0].timestamp() + (now - sent[1]) * sent[2]
            drift = abs(local.timestamp() - expected)
            if rate == sent[2] and drift < CLOCK_SLACK_S and now - sent[1] < CLOCK_EVERY_S:
                return
        self._clock_sent = (local, now, rate)
        item = {"clock": {"local": local.replace(tzinfo=None).isoformat(), "rate": rate}}
        with self._wake:
            heapq.heappush(self._due, (time.monotonic(), next(self._order), item))
            self._wake.notify()

    # ── what Duck Core does, played on the body ─────────────────────────────

    def after_call(
        self,
        session: AnySession,
        events_before: int,
        dues: dict[int, float] | None = None,
        fidgets: list[dict[str, Any]] | None = None,
        *,
        done_by_hand: dict[str, Any] | None = None,
    ) -> None:
        """Hand the motions of the events a call just produced to the physical body.

        ``dues`` gives each event's wall-clock moment (epoch seconds, by its index in the
        log): real time shows each event a moment after it happened, at the same moment as
        the page. Without it (something someone just did) they play now, after whatever was
        still waiting (it happened first) — or, when this call ends what came before
        (switching off, a pickup, a hand on its head, being put down, a fall, dozing off),
        instead of it. A hand event becomes the world's hand at its moment (see
        :func:`hand_command`) unless a family member's hand already did it (``done_by_hand``,
        a :class:`duckmind.simweb.family.Due`'s). Afterwards the body is asked for Duck Core's
        posture, so it never drifts from what the duck is doing.
        """
        if not self.ready:
            return
        if self.neuro is not None:
            self._hands_only(session, events_before, dues, done_by_hand)
            return
        wall, mono = time.time(), time.monotonic()
        nest = bool(getattr(session.core.hw, "nest", False))

        def at(epoch: float | None) -> float:
            return mono if epoch is None else mono + max(0.0, epoch - wall)

        with self._wake:
            events = session.core.log.events
            new = [
                (index, events[index], _motion(events[index]))
                for index in range(events_before, len(events))
            ]
            moves = [(index, event, motion) for index, event, motion in new if motion]
            if dues is None and moves:
                if any(event.kind in INTERRUPTS for _, event, _ in moves):
                    # (what the world's hand still has to do happened all the same: kept)
                    self._due = [item for item in self._due if "hand" in item[2]]
                    heapq.heapify(self._due)
                else:  # older moves first, now
                    waiting = sorted(self._due, key=lambda item: (item[0], item[1]))
                    self._due = [(mono, next(self._order), item[2]) for item in waiting]
                    heapq.heapify(self._due)
            last = mono
            hands: list[tuple[float, dict[str, Any]]] = []
            for index, event, motion in moves:
                due = at(dues.get(index) if dues else None)
                if event.kind in INTERRUPTS:
                    self._due = [item for item in self._due if item[0] < due or "hand" in item[2]]
                    heapq.heapify(self._due)
                heapq.heappush(self._due, (due, next(self._order), motion))
                last = max(last, due)
                command = hand_command(event, nest=nest)
                if command is None or done_already(command, done_by_hand):
                    continue
                if hands and hands[-1][1] == command:
                    continue  # (the same hand twice in a row: a pickup that puts it on its desk)
                hands.append((due, command))
            for due, command in hands:
                heapq.heappush(self._due, (due, next(self._order), {"hand": command}))
            for item in fidgets or []:
                motion = {key: item[key] for key in ("clip", "amplitude", "tempo", "weak")}
                heapq.heappush(self._due, (at(item["due_ms"] / 1000), next(self._order), motion))
            # Duck Core's posture now is where it will be once everything still waiting has
            # played (asked after new moves, and every few seconds anyway).
            want = _posture_of(session)
            if want is not None and (moves or mono - self._synced_at >= SYNC_EVERY_S):
                self._synced_at = mono
                self._want = want
                last = max([last, *(item[0] for item in self._due)])
                heapq.heappush(self._due, (last + 0.5, next(self._order), {"sync": want}))
            self._wake.notify()

    def _hands_only(
        self,
        session: AnySession,
        events_before: int,
        dues: dict[int, float] | None,
        done_by_hand: dict[str, Any] | None,
    ) -> None:
        """Prototype mode: of what a Duck Core call produced, only the world's hand (a pickup,
        a put-down, setting it upright — what the page's buttons do to the duck) reaches the
        world; the neural duck senses it through its own body. Duck Core's motions do not
        reach the body."""
        wall, mono = time.time(), time.monotonic()
        nest = bool(getattr(session.core.hw, "nest", False))
        with self._wake:
            events = session.core.log.events
            hands: list[tuple[float, dict[str, Any]]] = []
            for index in range(events_before, len(events)):
                command = hand_command(events[index], nest=nest)
                if command is None or done_already(command, done_by_hand):
                    continue
                if hands and hands[-1][1] == command:
                    continue
                epoch = dues.get(index) if dues else None
                due = mono if epoch is None else mono + max(0.0, epoch - wall)
                hands.append((due, command))
            for due, command in hands:
                heapq.heappush(self._due, (due, next(self._order), {"hand": command}))
            self._wake.notify()

    def resync(self, session: AnySession, why: str = "sync") -> None:
        """After time the physics did not live (a catch-up, a new duck) or a body that just
        became ready (``why`` "start"): drop what was waiting, put the body where Duck Core's
        plan is now (a world operation, :meth:`place`; a duck in its nest goes into the
        physical dock, :meth:`plan_pose`) and match Duck Core's power and posture."""
        if not self.ready or self.neuro is not None:
            return  # (prototype mode: Duck Core's plan does not move the neural duck's body)
        with self._sync_lock:
            self._sync_pending = None  # (this placement is newer than a waiting one)
        pose = self.plan_pose(session)
        self._sync(pose, _posture_of(session), bool(session.core.s.awake), why)

    def plan_pose(self, session: AnySession) -> tuple[float, float, float]:
        """Where a placement puts the body for Duck Core's state (world metres and radians):
        where its plan is — except a duck with a nest that sits in it, which goes into the
        physical charging place, seated as it docks (its plan's (0, 0) is the middle of the
        room in ``default_home``, not the dock: phase 5 teaches it the dock)."""
        core = session.core
        nest = getattr(self.layout, "nest", None)
        if nest is not None and bool(getattr(core.hw, "nest", False)) and core.s.nested:
            seat = nest.seat
            return seat.x, seat.y, seat.yaw
        return self.frame.to_world(core.pose_now())

    def _sync(
        self, pose: tuple[float, float, float], want: str | None, awake: bool, why: str
    ) -> None:
        with self._wake:
            self._due.clear()
        self.place(pose, why)
        with self._wake:
            now = time.monotonic()
            heapq.heappush(self._due, (now, next(self._order), {"power": "on" if awake else "off"}))
            if want is not None:
                self._want = want
                heapq.heappush(self._due, (now + 0.1, next(self._order), {"sync": want}))
            self._wake.notify()

    def skipped(self, session: AnySession) -> None:
        """A fast-forward step the physics did not live: once the steps stop (none for
        ``sync_quiet_s``), the body is put where Duck Core's plan then is ("sync")."""
        if not self.ready or self.neuro is not None:
            return  # (prototype mode: Duck Core's plan does not move the neural duck's body)
        pending = (
            self.plan_pose(session),
            _posture_of(session),
            bool(session.core.s.awake),
            time.monotonic(),
        )
        if self.sync_quiet_s <= 0:
            self._sync(pending[0], pending[1], pending[2], "sync")
            return
        with self._sync_lock:
            self._sync_pending = pending
            if self._sync_thread is None or not self._sync_thread.is_alive():
                self._sync_thread = threading.Thread(target=self._sync_later, daemon=True)
                self._sync_thread.start()

    def _sync_later(self) -> None:
        while not self._closing:
            with self._sync_lock:
                pending = self._sync_pending
                if pending is None:
                    return
                wait = pending[3] + self.sync_quiet_s - time.monotonic()
                if wait <= 0:
                    self._sync_pending = None
            if wait <= 0:
                if self.ready:
                    self._sync(pending[0], pending[1], pending[2], "sync")
                return
            time.sleep(min(wait, 0.25))

    def _play_due(self) -> None:
        while True:
            with self._wake:
                while not self._closing and (not self._due or self._due[0][0] > time.monotonic()):
                    timeout = self._due[0][0] - time.monotonic() if self._due else None
                    self._wake.wait(timeout)
                if self._closing:
                    return
                ready = []
                while self._due and self._due[0][0] <= time.monotonic():
                    ready.append(heapq.heappop(self._due)[2])
            moves = []
            held: list[dict[str, Any]] = []
            for item in ready:
                if "hand" in item:
                    if time.time() < self.demo_until:
                        held.append(item)  # (a demonstration has the body: the hand waits)
                    else:
                        self._hand(item["hand"])
                elif "clock" in item:
                    try:
                        self._call("x_lights", **item["clock"])
                    except Exception:
                        self._clock_sent = None  # (tried again with the next second)
                else:
                    moves.append(item)
            if held:
                after = time.monotonic() + max(0.0, self.demo_until - time.time()) + 0.2
                with self._wake:
                    for item in held:
                        heapq.heappush(self._due, (after, next(self._order), item))
            if time.time() < self.demo_until:
                # A demonstration has the body: only Duck Core's head moves and power reach it.
                moves = [
                    item for item in moves if "clip" in item or "power" in item or "wake" in item
                ]
            if self.ready and moves:
                with self._lock:
                    self.body.consume(moves)

    # ── physics demonstrations (never Duck Core) ────────────────────────────

    def demo_walk(self) -> None:
        """A physics demonstration outside Duck Core: stand up and walk one short loop."""
        if not self.ready:
            raise ValueError("the physics body is still starting")
        if self.neuro is not None:
            raise ValueError(NEURO_BUSY_ZH)
        with self._wake:
            self._due.clear()  # the demonstration has the body to itself
            self._synced_at = time.monotonic() + DEMO_S + 3.0  # then back to Duck Core's posture
        self.demo_until = time.time() + DEMO_S + 3.0
        with self._lock:
            self.body.consume([{"walk": {"vx": 0.35, "vyaw": 0.4, "seconds": 6.0}}])

    def demo(self, kind: str, thing: str, side: str | None = None) -> dict[str, Any]:
        """示範 (curiosity P0): bump, peck or kick ``thing`` — the duck walks over, the world
        puts the thing where the official skill meets it (truth positioning, 示範 only), the
        skill runs and the world says what happened (:mod:`duckmind.simweb.demo`)."""
        from duckmind.simweb.demo import LABEL_ZH, DemoRunner

        if not self.world_ready():
            raise ValueError(msg("web.live.LiveBridge.demo.363966"))
        if self.neuro is not None:
            raise ValueError(NEURO_BUSY_ZH)
        if self._demo is not None:
            raise ValueError(msg("web.live.LiveBridge.demo.bd6d03"))
        runner = DemoRunner(kind, thing, self.world_port(), side=side)
        self._demo = runner
        self.demo_result = None
        with self._wake:
            self._due.clear()
        self.demo_until = time.time() + DEMO_LIMIT_S + 5.0
        threading.Thread(target=self._run_demo, args=(runner,), daemon=True).start()
        return {
            "kind": kind,
            "thing": thing,
            "zh": msg("web.live.LiveBridge.demo.ededc8", kind=LABEL_ZH[kind]),
        }

    def _run_demo(self, runner: Any) -> None:
        began = time.monotonic()
        result: dict[str, Any] = {}
        try:
            while not self._closing and self._demo is runner:
                command = runner.tick(time.monotonic(), policy=getattr(self.body, "policy", None))
                self._demo_command(command)
                if "done" in command:
                    result = dict(command.get("done") or {})
                    break
                if time.monotonic() - began > DEMO_LIMIT_S:
                    runner.why_zh.append(msg("web.live.LiveBridge.run_demo.303208"))
                    result = dict(runner._finish(time.monotonic(), ok=False)["done"])
                    break
                time.sleep(TICK_S)
        except Exception as error:
            result = {
                "ok": False,
                "demo": True,
                "why_zh": [msg("web.live.LiveBridge.run_demo.3c50b7", error=str(error))],
            }
        finally:
            if self._demo is runner:
                self._demo = None
                self.demo_result = {**result, "at_ms": int(time.time() * 1000)}
                self.demo_until = time.time() + 0.5
                want = self._want
                if want is not None and self.ready:
                    with self._wake:
                        heapq.heappush(
                            self._due, (time.monotonic() + 0.6, next(self._order), {"sync": want})
                        )
                        self._wake.notify()

    def eye_demo(self, kind: str, thing: str) -> dict[str, Any]:
        """「用眼睛做」 (owner decision A6): one curiosity step through the duck-body and duck-eye
        processes (:class:`duckmind.simweb.eyedemo.EyeRun`). The world only puts ``thing`` on the
        floor about half a metre in front of the duck; the duck finds it with its own eyes, walks
        up, aligns and acts. Duck Core is not told."""
        from duckmind.simweb.eyedemo import VERBS, EyeRun, Sockets, floor_file

        if self.neuro is not None:
            raise ValueError(NEURO_BUSY_ZH)
        if kind not in VERBS:
            raise ValueError(msg("web.live.LiveBridge.eye_demo.453f25"))
        if not self.world_ready():
            raise ValueError(msg("web.live.LiveBridge.eye_demo.21c7d7"))
        if self._demo is not None:
            raise ValueError(msg("web.live.LiveBridge.eye_demo.bd6d03"))
        body = self.body
        robotd = getattr(body, "robotd_socket", None)
        run = getattr(body, "state_dir", None)
        if robotd is None or run is None:
            raise ValueError(msg("web.live.LiveBridge.eye_demo.d69176"))
        layout = self.layout
        items = {} if layout is None else {t["id"]: t for t in layout.to_json().get("things", [])}
        item = items.get(thing)
        if item is None:
            raise ValueError(msg("web.live.LiveBridge.eye_demo.114c28"))
        verbs = item.get("verbs") or []
        if verbs and kind not in verbs:
            raise ValueError(
                msg("web.live.LiveBridge.eye_demo.2ff92b", value=item.get("zh") or thing)
            )
        _camera, media, why = self._media_socket(body)
        if media is None:
            raise ValueError(why or msg("web.live.LiveBridge.eye_demo.6f7677"))
        run_dir = Path(run) / "phase3"
        mjcf = official_mjcf(self.paths)
        sockets = Sockets(
            str(robotd),
            media,
            None,
            run_dir,
            mjcf / "robot_allcollisions.xml",
            Path(self.paths.microduck) / "odometry" / "src" / "anchors.rs",
            floor_model=floor_file(self.home, self.eye_dir or run_dir),
            dock_style=getattr(getattr(layout, "nest", None), "style", None),
            dock_side=getattr(getattr(layout, "nest", None), "side", None) or None,
        )
        runner = EyeRun(
            kind, item, world=self._call, truth=self.truth_now, body=body, sockets=sockets
        )
        self._demo = runner
        self.demo_result = None
        with self._wake:
            self._due.clear()
        self.demo_until = time.time() + EYE_DEMO_LIMIT_S
        threading.Thread(target=self._run_eye_demo, args=(runner,), daemon=True).start()
        return {"kind": kind, "thing": thing, "zh": runner.zh}

    def _run_eye_demo(self, runner: Any) -> None:
        result: dict[str, Any] = {}
        try:
            result = runner.run()
        except Exception as error:  # the page says why; the processes stop below
            result = {
                "ok": False,
                "demo": True,
                "eye": True,
                "zh": runner.zh,
                "why_zh": [msg("web.live.LiveBridge.run_eye_demo", error=str(error))],
            }
        finally:
            runner.close()
            if self._demo is runner:
                self._demo = None
                self.demo_result = {**result, "at_ms": int(time.time() * 1000)}
                self.demo_until = time.time() + 0.5
                want = self._want
                if want is not None and self.ready:
                    with self._wake:
                        heapq.heappush(
                            self._due, (time.monotonic() + 0.6, next(self._order), {"sync": want})
                        )
                        self._wake.notify()

    def _demo_command(self, command: dict[str, Any]) -> None:
        body = self.body
        if "move" in command and hasattr(body, "drive"):
            vx, vy, vyaw = (float(value) for value in command["move"])
            body.drive(vx, vy, vyaw)
        if command.get("posture") in ("stand", "sit"):
            body.request_posture(command["posture"])
        if command.get("skill") and hasattr(body, "skill"):
            body.skill(command["skill"])

    def _stop_demo(self) -> None:
        runner, self._demo = self._demo, None
        close = getattr(runner, "close", None)
        if callable(close):  # 「用眼睛做」: its duck-body and duck-eye processes stop too
            close()

    def demo_view(self) -> dict[str, Any] | None:
        runner = self._demo
        if runner is not None:
            return {
                "running": True,
                "kind": runner.kind,
                "thing": runner.thing,
                "state": runner.state,
                "why_zh": list(runner.why_zh),
                "eye": getattr(runner, "zh", "").startswith("用眼睛做"),
            }
        if self.demo_result is not None:
            return {"running": False, **self.demo_result}
        return None

    # ── what the page reads ─────────────────────────────────────────────────

    def world_view(self) -> dict[str, Any]:
        """The world for the page: the home it runs (``home``: id, variant, source,
        fingerprint; ``physical`` when the physics has the home's walls and furniture), Duck
        Core's origin in it (``frame``), the last placement and hand request, and — from the
        world's truth (x_world) — the hand, the people and their bodies, the things, the world
        battery (display only, E1), the nest's contact and the backlight."""
        state = self._world() or {}
        home = state.get("home") or {}
        out: dict[str, Any] = {
            "home": home or None,
            "asked": self.home,
            "physical": home.get("source") == "layout",
            "frame": self.frame.view(),
            "placed": self._placed,
            "hand_note": self.hand_note,
            "demo": self.demo_view(),
            "sim_time": state.get("sim_time"),
        }
        hand = state.get("hand")
        if isinstance(hand, dict):
            out["hand"] = {
                key: hand.get(key)
                for key in (
                    "state",
                    "do",
                    "holding",
                    "shown",
                    "pos_mm",
                    "quat",
                    "target",
                    "eta_s",
                    "why_zh",
                    "_i18n",  # (the world's words' messages: static/i18n.js)
                )
                if key in hand or key != "_i18n"
            }
        people = state.get("people")
        present: set[str] = set()
        if isinstance(people, list):
            out["people"] = [
                {
                    key: person.get(key)
                    for key in (
                        "who",
                        "zh",
                        "state",
                        "pos_mm",
                        "heading_mrad",
                        "clearance_mm",
                        "eta_in_s",
                        "_i18n",
                    )
                    if key in person or key != "_i18n"
                }
                for person in people
                if isinstance(person, dict)
            ]
            present = {str(person.get("who")) for person in people if isinstance(person, dict)}
        bodies: dict[str, Any] = {}
        for item in state.get("mocap") or []:
            name = str(item.get("name") or "")
            if name.startswith("person_") and name[7:].split("_leg")[0] in present:
                bodies[name] = {"pos_mm": item.get("pos_mm"), "quat": item.get("quat")}
        out["bodies"] = bodies
        things = state.get("things")
        if isinstance(things, list):
            out["things"] = [
                {key: thing.get(key) for key in ("id", "pos_mm", "quat", "moved_mm", "speed_mm_s")}
                for thing in things
                if isinstance(thing, dict)
            ]
        battery = state.get("battery")
        if isinstance(battery, dict) and "percent" in battery:
            out["battery"] = {
                key: battery.get(key)
                for key in (
                    "percent",
                    "volts_mv",
                    "current_ma",
                    "charging",
                    "pg",
                    "fed_to_robotd",
                    "empty",
                )
            }
        nest = state.get("nest")
        if isinstance(nest, dict):
            out["nest"] = {
                key: nest.get(key)
                for key in (
                    "present",
                    "kind",
                    "style",
                    "contact",
                    "pg",
                    "charging",
                    "pressed_mm",
                    "powered",
                    "probes",
                    "bumper",
                )
            }
        lights = state.get("lights")
        if isinstance(lights, dict):
            out["backlight"] = lights.get("backlight")
        return out

    def frames_since(self, since_t: float | None = None) -> list[dict[str, Any]]:
        """Every body sample the page does not have yet (``since_t``: the newest it has, in
        simulated seconds; see :meth:`duckmind.livebody.LiveBody.frames_since`), so the 3D view
        plays the physics at its true speed instead of easing toward the latest frame."""
        if not self.ready:
            return []
        take = getattr(self.body, "frames_since", None)
        return list(take(since_t)) if callable(take) else []

    def snapshot(self, since_t: float | None = None, *, frames: bool = False) -> dict[str, Any]:
        """The live body for the page: its state, the stall monitor (``monitor``: trust and
        why), the latest frame with its ``truth`` in the world, the world (``world``, see
        :meth:`world_view`) and the duck's-eye status. ``error`` is the program's own words,
        ``error_zh`` the same in plain words. ``frames``: also every body sample after
        ``since_t`` (``frames``, compact: ``t``, ``w``, ``j`` in ``frame_joints`` order, ``p``,
        ``q``, ``m``) for a page that plays them all."""
        body = self.body
        status = body.status()
        frame = self._latest_frame()
        if frame is not None:
            frame = {**frame, "truth": self.truth(frame)}
        error = self.error or body.error
        out = {
            "available": self.ready and not body.error,
            "starting": not self.ready and self.error is None,
            "error": error,
            "error_zh": plain_error(error),
            "realtime": status["realtime"],
            "policy": status["policy"],
            "fallen": status["fallen"],
            "held": bool(status.get("held")),
            "monitor": status.get("monitor"),
            "frame": frame,
            "world": self.world_view(),
            "eye": self.eye_status(),
            "demo": time.time() < self.demo_until,
        }
        if frames:
            out["frames"] = self.frames_since(since_t)
            out["frame_joints"] = list(FRAME_JOINTS)
            # (prototype mode: the samples carry the beak and the calls, ``m`` and ``v``)
            out["frame_calls"] = self.neuro is not None
        if self.neuro is not None:
            out["neuro"] = self.neuro.view()  # prototype mode only
        return out

    def stop(self) -> None:
        with self._wake:
            self._closing = True
            self._wake.notify()
        if self.neuro is not None:
            self.neuro.stop()
        self._stop_demo()
        self._stop_eye(wait=True)
        self.body.stop()
        for retiring in self._retiring:  # (an old body still being stopped after a restart)
            retiring.join(timeout=30)
