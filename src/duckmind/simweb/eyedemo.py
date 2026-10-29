"""「用眼睛做」 (owner decision A6, phase 3): one curiosity step done by the duck's own eyes and body.

The 示範 buttons of card 09 (:mod:`duckmind.simweb.demo`) put the thing on the trained spot by
truth and play the official skill. This one runs the phase-3 processes instead, the way the real
duck will run them:

* ``python -m duckmind.eye.server`` — duck-eye, serving ``eye.sock``;
* ``python -m duckmind.duckbody`` — duck-body: robotd's client (``robotd.sock`` of the live body),
  one ``media.sock`` connection handed to duck-eye per look, serving ``body.sock``;
* this module, as the only ``body.sock`` client, sends ``duck.posture`` (stand) and then one
  ``duck.curio`` step (:func:`duckmind.curiosity.demo.eye_demo_command`) and reads the percepts
  back until the step's ``done``.

The world only sets the scene: it puts the thing on the floor about half a metre in front of the
duck (roughly — not on the trained spot). Finding it, walking up, aligning and acting are the
duck's own (the Act Runner through duck-body's reflexes and guard; the eye's ``things`` task).
Duck Core is not told and does not remember it. The processes run only for the demonstration:
while they do, the live body sends robotd nothing (``LiveBody.passive``), because duck-body is
then robotd's only writer (spec §6).

Threads and sockets live here and in the two processes; the step itself is the pure Act Runner.
"""

from __future__ import annotations

import json
import os
import random
import select
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckmind
from duckmind.i18n import msg

VERBS = ("kick", "peck", "bump")  # grab waits for the jaw in the live world (owner question)
START_TIMEOUT_S = 30.0  # both processes up and connected
STAND_TIMEOUT_S = 12.0  # duck.posture stand (spec §5.2: 6 s, retried)
LOOK_TIMEOUT_S = 12.0  # the first look (light and what is ahead)
TEACH_WALKS = 3  # at most this many short walks to learn the floor (owner D1) …
TEACH_WALK_S = 12.0  # … each within this
READY_WALKS, READY_SAMPLES = 3, 4000  # duck-eye's floor model is trusted from here (FL-v1)
STEP_LIMIT_S = 100.0  # the curiosity step's own deadline (normal patience; median 14–27 s)
PLACE_MM = (450, 560)  # where the world puts the thing: this far ahead of the trunk …
PLACE_SIDE_MM = 60  # … and at most this far to either side
NOTE_ZH = msg("web.eyedemo.NOTE_ZH")


def monotonic_ms() -> int:
    """duck-body's clock (``CLOCK_MONOTONIC``, spec §7.1 — not ``time.monotonic`` on macOS)."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC) // 1_000_000


def colour_label(hex_colour: str | None) -> str | None:
    """What duck-eye's ``things`` task calls a colour (``#f07a1f`` → ``orange``): the label the
    step looks for when the toy is not registered with the eye."""
    if not hex_colour or not str(hex_colour).startswith("#") or len(str(hex_colour)) != 7:
        return None
    from duckmind.eye.things import colour_name

    r, g, b = (int(str(hex_colour)[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    top, low = max(r, g, b), min(r, g, b)
    d = top - low
    sat = d / top if top > 1e-6 else 0.0
    if d <= 1e-6:
        hue = 0.0
    elif top == r:
        hue = ((g - b) / d) % 6.0
    elif top == g:
        hue = (b - r) / d + 2.0
    else:
        hue = (r - g) / d + 4.0
    return colour_name(hue * 60.0, sat, top)


def place_ahead(pose: Mapping[str, Any], rng: random.Random) -> tuple[int, int]:
    """A floor spot roughly in front of the duck (world mm) for the thing."""
    import math

    yaw = float(pose["yaw_mrad"]) / 1000.0
    ahead = rng.uniform(*PLACE_MM)
    side = rng.uniform(-PLACE_SIDE_MM, PLACE_SIDE_MM)
    x = float(pose["x_mm"]) + ahead * math.cos(yaw) - side * math.sin(yaw)
    y = float(pose["y_mm"]) + ahead * math.sin(yaw) + side * math.cos(yaw)
    return int(round(x)), int(round(y))


FLOORS = Path(__file__).resolve().parent / "floors"  # floors a simulated duck already learnt


def floor_file(home: str | None, folder: Path) -> Path:
    """The duck's learned floor for ``home`` in ``folder`` (kept between demonstrations). The
    first time, a floor this home's simulated duck already learnt is copied in when there is one
    (``simweb/floors/<home>.json``: FL-v1 learned from 24 walked-clear corridors in headless
    renders of the home, eye-vision lane, phase 3) — a duck that has walked here before.
    Otherwise the eye starts unlearned and names no things until it has walked."""
    name = (home or "flat").split(",")[0]
    path = folder / f"{name}.floor.json"
    seed = FLOORS / f"{name}.json"
    if not path.exists() and seed.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(seed.read_bytes())
    return path


def floor_ready(path: Path | None) -> bool:
    """Whether the learned floor in ``path`` is trusted yet (FL-v1: 3 walks, 4000 samples);
    without it duck-eye names no things on the floor."""
    if path is None or not path.is_file():
        return False
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    walks, samples = model.get("walks", 0), model.get("samples", 0)
    return (
        isinstance(walks, int)
        and isinstance(samples, int)
        and (walks >= READY_WALKS and samples >= READY_SAMPLES)
    )


@dataclass
class Sockets:
    """Where the live body's doors are and where the two processes put theirs."""

    robotd: str
    media: str | None
    tof: str | None
    run_dir: Path
    mjcf_xml: Path | None = None
    anchors: Path | None = None
    floor_model: Path | None = None  # the duck's learned floor (kept between demonstrations)
    dock_style: str | None = None  # the home's dock (forward | backin): duck-body's duck.dock
    dock_side: str | None = None  # backin's turning side (right | left: the home's nest.side)

    @property
    def eye(self) -> str:
        return str(self.run_dir / "eye.sock")

    @property
    def body(self) -> str:
        return str(self.run_dir / "body.sock")


class BodyLink:
    """``body.sock``'s client end: JSON lines in and out."""

    def __init__(self, path: str, timeout_s: float = 2.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect(path)
        self.buffer = bytearray()

    def send(self, obj: Mapping[str, Any]) -> None:
        self.sock.sendall(json.dumps(obj, separators=(",", ":")).encode() + b"\n")

    def recv(self, timeout_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                line = bytes(self.buffer[:end])
                del self.buffer[: end + 1]
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        return value
                continue
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self.sock], [], [], left)[0]:
                return None
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError(msg("web.eyedemo.BodyLink.recv"))
            self.buffer.extend(data)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


@dataclass
class EyeDemo:
    """One 「用眼睛做」 run: start duck-eye and duck-body, stand, one ``duck.curio`` step, stop.

    ``kind``/``thing``/``state``/``why_zh`` are what the page's demo view shows while it runs
    (the same fields as :class:`duckmind.simweb.demo.DemoRunner`)."""

    kind: str
    thing: str
    sockets: Sockets
    python: str = sys.executable
    state: str = "starting"
    why_zh: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    procs: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict)
    link: BodyLink | None = None
    hello: dict[str, Any] | None = None
    stopping: bool = False

    # ── the processes ──

    def _env(self) -> dict[str, str]:
        root = str(Path(duckmind.__file__).resolve().parent.parent)
        path = os.environ.get("PYTHONPATH")
        return {
            **os.environ,
            "PYTHONPATH": root if not path else os.pathsep.join((root, path)),
            "DUCKMIND_BODY_LOG_LOOKS": "1",  # every look in duck-body's log (its run folder)
        }

    def _spawn(self, name: str, args: list[str]) -> subprocess.Popen[bytes]:
        log = (self.sockets.run_dir / f"{name}.log").open("wb")
        proc = subprocess.Popen(
            [self.python, "-m", *args],
            env=self._env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        self.procs[name] = proc
        return proc

    def _wait_socket(self, name: str, path: str, deadline: float) -> None:
        proc = self.procs[name]
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} stopped: {self.log_tail(name)}")
            if os.path.exists(path):
                return
            time.sleep(0.1)
        raise TimeoutError(f"{name} did not open {path}")

    def log_tail(self, name: str, lines: int = 4) -> str:
        try:
            text = (self.sockets.run_dir / f"{name}.log").read_text(errors="replace")
        except OSError:
            return ""
        return " | ".join(text.strip().splitlines()[-lines:])

    def start(self, timeout_s: float = START_TIMEOUT_S) -> None:
        s = self.sockets
        s.run_dir.mkdir(parents=True, exist_ok=True)
        for path in (s.eye, s.body):
            if os.path.exists(path):
                os.unlink(path)
        deadline = time.monotonic() + timeout_s
        if s.media is not None:
            eye = ["duckmind.eye.server", "--socket", s.eye]
            if s.floor_model is not None:
                eye += ["--floor-model", str(s.floor_model)]
            self._spawn("duck-eye", eye)
            self._wait_socket("duck-eye", s.eye, deadline)
        args = ["duckmind.duckbody", "--robotd", s.robotd, "--body", s.body, "--src", "live"]
        if s.mjcf_xml is not None:
            args += ["--mjcf", str(s.mjcf_xml)]
            if s.anchors is not None:
                args += ["--anchors", str(s.anchors)]
        if s.media is not None:
            args += ["--eye", s.eye, "--media", s.media]
        if s.tof is not None:
            args += ["--tof", s.tof]
        if s.dock_style in ("forward", "backin"):
            args += ["--dock-style", s.dock_style]
        if s.dock_side in ("right", "left"):
            args += ["--dock-side", s.dock_side]  # the apartment's mirrored J-hook is left
        self._spawn("duck-body", args)
        self._wait_socket("duck-body", s.body, deadline)
        while time.monotonic() < deadline:
            try:
                self.link = BodyLink(s.body)
                break
            except OSError:
                time.sleep(0.1)
        if self.link is None:
            raise TimeoutError("could not connect to body.sock")
        self.hello = self.link.recv(max(0.5, deadline - time.monotonic()))
        if not self.hello or self.hello.get("type") != "hello":
            raise RuntimeError(f"duck-body said no hello: {self.hello}")

    def stop(self) -> None:
        self.stopping = True
        if self.link is not None:
            self.link.close()
            self.link = None
        for name in ("duck-body", "duck-eye"):  # duck-body first: it lets go of robotd
            proc = self.procs.get(name)
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

    # ── acts over body.sock ──

    def act(
        self,
        command: Any,
        limit_s: float,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[bool, str | None, list[dict[str, Any]]]:
        """Send one act and read until its ``done``: (accepted, refusal reason, its percepts)."""
        from duckmind.duckbody.protocol import command_to_wire

        link = self.link
        if link is None:
            raise RuntimeError(msg("web.eyedemo.EyeDemo.act"))
        link.send(command_to_wire(command))
        act_id = command.act_id
        mine: list[dict[str, Any]] = []
        deadline = time.monotonic() + limit_s
        while time.monotonic() < deadline and not self.stopping:
            line = link.recv(0.5)
            if line is None:
                continue
            kind = line.get("type")
            if kind == "ack" and line.get("act_id") == act_id and not line.get("ok"):
                return False, str(line.get("reason") or "refused"), mine
            if kind == "percept":
                event = line.get("event")
                if isinstance(event, dict):
                    self.events.append(event)
                    if event.get("act_id") == act_id:
                        mine.append(event)
                    if on_event is not None:
                        on_event(event)
            if kind == "done" and line.get("act_id") == act_id:
                return True, None, mine
        return True, "timeout", mine


REFUSED_ZH: dict[str, str] = {
    "lifted": msg("web.eyedemo.REFUSED_ZH.lifted"),
    "fallen": msg("web.eyedemo.REFUSED_ZH.fallen"),
    "righting": msg("web.eyedemo.REFUSED_ZH.righting"),
    "busy": msg("web.eyedemo.REFUSED_ZH.busy"),
    "unavailable": msg("web.eyedemo.REFUSED_ZH.unavailable"),
    "past_deadline": msg("web.eyedemo.REFUSED_ZH.past_deadline"),
    "invalid": msg("web.eyedemo.REFUSED_ZH.invalid"),
    "forbidden": msg("web.eyedemo.REFUSED_ZH.forbidden"),
    "timeout": msg("web.eyedemo.REFUSED_ZH.timeout"),
}


class EyeRun:
    """The 「用眼睛做」 run the page's demo slot holds: set the scene, start the processes, stand,
    one ``duck.curio`` step, report — :meth:`run` on a thread, :meth:`close` always after.

    ``world(op, **params)`` is the live body's world port (``x_place_prop``, ``x_sensors``);
    ``truth()`` where the physics body is (only to put the thing in front of it); ``body`` the
    :class:`~duckmind.livebody.LiveBody` (``passive`` while duck-body drives)."""

    def __init__(
        self,
        kind: str,
        item: Mapping[str, Any],
        *,
        world: Callable[..., dict[str, Any]],
        truth: Callable[[], Mapping[str, Any] | None],
        body: Any,
        sockets: Sockets,
        seed: int | None = None,
    ) -> None:
        from duckmind.curiosity.demo import VERBS_ZH

        self.kind = kind
        self.item = dict(item)
        self.thing = str(item.get("id"))
        self.thing_zh = str(item.get("zh") or self.thing)
        self.zh = msg("web.eyedemo.EyeRun.init", kind=VERBS_ZH.get(kind, kind))
        self.world = world
        self.truth = truth
        self.body = body
        self.demo = EyeDemo(kind, self.thing, sockets)
        self.rng = random.Random(seed)
        self.state = "starting"
        self.why_zh: list[str] = []
        self.placed: dict[str, Any] | None = None
        self.teach: list[dict[str, Any]] = []

    def _fail(self, said: str, **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "demo": True,
            "eye": True,
            "zh": self.zh,
            "kind": self.kind,
            "thing": self.thing,
            "why_zh": [said],
            "note_zh": NOTE_ZH,
            **extra,
        }

    def run(self) -> dict[str, Any]:
        from duckmind.curiosity.act_runner import ANY_THING
        from duckmind.curiosity.demo import describe_zh, eye_demo_command
        from duckmind.duckbody.protocol import LookCmd, PostureCmd

        self.state = "starting"
        self.body.passive = True
        time.sleep(0.3)
        self.demo.start()
        self.state = "standing"
        stand = PostureCmd(act_id="P1", deadline_ms=monotonic_ms() + 15_000, want="stand")
        ok, why, events = self.demo.act(stand, STAND_TIMEOUT_S + 3.0)
        got = next((e.get("got") for e in events if e.get("kind") == "body.posture"), None)
        if not ok or got != "stand":
            said = (
                REFUSED_ZH.get(str(why), msg("web.eyedemo.EyeRun.run.310296"))
                if why
                else msg("web.eyedemo.EyeRun.run.310296")
            )
            return self._fail(msg("web.eyedemo.EyeRun.run.135412", said=said))
        if not floor_ready(self.demo.sockets.floor_model):
            # Owner D1: the eye knows the floor only from ground the duck walked safely; a duck
            # that has not walked here yet first takes a few short arcs to learn it.
            self.state = "learning"
            self.teach = self._teach()
            if not floor_ready(self.demo.sockets.floor_model):
                return self._fail(
                    msg("web.eyedemo.EyeRun.run.6ac9a5"),
                    teach=self.teach,
                )
        pose = self.truth()
        if pose is None:
            return self._fail(msg("web.eyedemo.EyeRun.run.8eac11"))
        x, y = place_ahead(pose, self.rng)
        self.placed = self.world(
            "x_place_prop", id=self.thing, x_mm=x, y_mm=y, yaw_mrad=0, on="floor"
        )
        if isinstance(self.placed, Mapping) and self.placed.get("ok") is False:
            return self._fail(msg("web.eyedemo.EyeRun.run.dc8b5d"))
        time.sleep(0.5)
        # A first look: the light (no curiosity act starts in the dark, or before the eye has
        # said how bright it is) and what is in front.
        self.state = "looking"
        look = LookCmd(
            act_id="L1", deadline_ms=monotonic_ms() + 10_000, kind="fixate", want="thing"
        )
        ok, why, events = self.demo.act(look, LOOK_TIMEOUT_S)
        scan = next((e for e in events if e.get("kind") == "sense.scan"), None)
        if not ok or scan is None:
            said = (
                REFUSED_ZH.get(str(why), msg("web.eyedemo.EyeRun.run.b3e4a3"))
                if why
                else msg("web.eyedemo.EyeRun.run.b3e4a3")
            )
            return self._fail(msg("web.eyedemo.EyeRun.run.a51464", said=said))
        self.state = "doing"
        command = eye_demo_command(
            self.item,
            self.kind,
            act_id="C1",
            deadline_ms=monotonic_ms() + int(STEP_LIMIT_S * 1000),
            label=ANY_THING,
        )
        # What the duck is told is only what the family would say — "in front of you, about
        # half a metre" (straight ahead, the middle of where the world puts it) — never where the
        # world put it; the eye finds it there by place, whatever colour name it gives it.
        command.params.update(b_mrad=0, r_mm=(PLACE_MM[0] + PLACE_MM[1]) // 2)
        ok, why, events = self.demo.act(command, STEP_LIMIT_S + 5.0)
        step = next((e for e in events if e.get("kind") == "body.curio_step"), None)
        if step is None:
            said = (
                REFUSED_ZH.get(str(why), str(why)) if why else msg("web.eyedemo.EyeRun.run.6e6128")
            )
            return self._fail(
                msg("web.eyedemo.EyeRun.run.c6eb57", said=said), percepts=len(self.demo.events)
            )
        keep = ("outcome", "ended_by", "ms", "looks", "looks_lost", "moved_mm", "light", "fell")
        return {
            "ok": bool(step.get("ok")),
            "demo": True,
            "eye": True,
            "zh": self.zh,
            "kind": self.kind,
            "thing": self.thing,
            "why_zh": [describe_zh(step, self.thing_zh)],
            "note_zh": NOTE_ZH,
            "step": {key: step.get(key) for key in keep if key in step},
            "percepts": len(self.demo.events),
        }

    def _teach(self) -> list[dict[str, Any]]:
        """Short walks (0.40 m/s arcs to the left, a glance before each) until duck-eye's floor
        model is trusted: each clear walk's ground goes to duck-eye as ``eye.learn``."""
        from duckmind.duckbody.protocol import WalkCmd

        done: list[dict[str, Any]] = []
        for n in range(1, TEACH_WALKS + 1):
            walk = WalkCmd(
                act_id=f"W{n}",
                deadline_ms=monotonic_ms() + int(TEACH_WALK_S * 1000),
                vx_mm=400,
                vyaw_mrad=500,
                max_ms=2000,
                pre_look="glance",
                purpose="wander",
            )
            ok, why, events = self.demo.act(walk, TEACH_WALK_S + 3.0)
            end = next((e for e in events if e.get("kind") == "walk.done"), {})
            done.append(
                {"ok": ok, "why": why, "reason": end.get("reason"), "odom": end.get("odom")}
            )
            if not ok:
                break
            time.sleep(0.3)  # duck-eye writes the model after the eye.learn
            if floor_ready(self.demo.sockets.floor_model):
                break
        return done

    def close(self) -> None:
        try:
            self.demo.stop()
        finally:
            self.body.passive = False


__all__ = [
    "BodyLink",
    "EyeDemo",
    "EyeRun",
    "REFUSED_ZH",
    "NOTE_ZH",
    "Sockets",
    "VERBS",
    "colour_label",
    "monotonic_ms",
    "place_ahead",
]
