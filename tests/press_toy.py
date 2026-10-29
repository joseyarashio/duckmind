"""A kinematic toy for closed-loop tests of getting stuck: the whole neural brain, the real body
link (duck-body's proprio, stall holds, ToF verdicts) and a small room with walls and boxes.

NOT the simulator and NOT physics (the headless MuJoCo measurement is
``tools/machine_verify/measure_unstuck.py``): the body goes where the twist the link let through
says (the rough kinematics of ``duckmind.neuro.synth``: 0.31 × vx forward, 0.25 × vx back, yaw rate
0.6 × vyaw), cannot go into a wall or a box, and slides along it. The senses are the ones a duck has
in front of a wall: a ToF that sees walls and tall boxes (a *hidden* box is under its beam: felt,
not seen), contact odometry, and a gentle touch at the onset of a contact. Truth (the pose, the
gap to what it is touching) is read only by the tests.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome
from duckmind.simweb.neuro_live import _events_of

HOME_JOINTS = (0.3491, 0.3491, 0.0, 0.0)
DT = 0.02
NOSE = 0.10  # trunk centre to nose
TAIL = 0.07
SENSOR = 0.08  # the ToF sits this far ahead of the trunk centre


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    hidden: bool = False  # low: under the ToF's beam (felt, not seen)


@dataclass(frozen=True)
class Row:
    t: float
    x: float
    y: float
    yaw: float
    vx: float  # what the body link let through
    vyaw: float
    want_vx: float  # what the brain asked for
    want_vyaw: float
    tof_ahead: int | None
    behaviour: str
    reflex: str | None
    gate: tuple[str, ...]
    gap: float  # truth: nose to what is ahead of it
    lean_in: bool


class PressToy:
    """One neural duck and its body link in a room [0, W] x [0, H] with boxes."""

    def __init__(
        self,
        brain: NeuroBrain,
        *,
        x: float,
        y: float,
        yaw: float,
        W: float = 4.0,
        H: float = 4.0,
        boxes: tuple[Box, ...] = (),
    ) -> None:
        self.brain = brain
        self.W, self.H, self.boxes = W, H, list(boxes)
        self.x, self.y, self.yaw = x, y, yaw
        self.x0, self.y0, self.yaw0 = x, y, yaw
        self.link = BodyLink()
        self.t = 0.0
        self.last: Any = None
        self.applied = (0.0, 0.0)
        self.odom = (0.0, 0.0, 0.0)
        self.rows: list[Row] = []
        self._contact_since: float | None = None

    # ── truth ────────────────────────────────────────────────────────────────

    def _inside(self, px: float, py: float, *, seen_only: bool = False) -> bool:
        if not (0.0 <= px <= self.W and 0.0 <= py <= self.H):
            return False
        for b in self.boxes:
            if seen_only and b.hidden:
                continue
            if b.x0 < px < b.x1 and b.y0 < py < b.y1:
                return False
        return True

    def _points(self, x: float, y: float, yaw: float) -> list[tuple[float, float]]:
        c, s = math.cos(yaw), math.sin(yaw)
        return [
            (x + NOSE * c, y + NOSE * s),
            (x + 0.04 * c - 0.06 * s, y + 0.04 * s + 0.06 * c),
            (x + 0.04 * c + 0.06 * s, y + 0.04 * s - 0.06 * c),
            (x - TAIL * c, y - TAIL * s),
        ]

    def _ok(self, x: float, y: float, yaw: float) -> bool:
        return all(self._inside(px, py) for px, py in self._points(x, y, yaw))

    def _ray(
        self, ox: float, oy: float, ang: float, far: float = 4.0, seen_only: bool = True
    ) -> float:
        c, s = math.cos(ang), math.sin(ang)
        for i in range(1, int(far / 0.01) + 1):
            r = i * 0.01
            if not self._inside(ox + r * c, oy + r * s, seen_only=seen_only):
                return r
        return far

    def nose_gap(self) -> float:
        """Truth: how far ahead of the nose anything (a hidden box too) is, up to 0.3 m."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self._ray(self.x + NOSE * c, self.y + NOSE * s, self.yaw, far=0.3, seen_only=False)

    def _tof(self) -> tuple[int | None, int | None, int | None]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ox, oy = self.x + SENSOR * c, self.y + SENSOR * s

        def mm(r: float) -> int | None:
            return None if r >= 1.5 else int(r * 1000)

        return (
            mm(self._ray(ox, oy, self.yaw + 0.5)),
            mm(self._ray(ox, oy, self.yaw)),
            mm(self._ray(ox, oy, self.yaw - 0.5)),
        )

    # ── the loop ─────────────────────────────────────────────────────────────

    def _state(self) -> RobotState:
        joints = [0.0] * 15
        for i, home in enumerate(HOME_JOINTS):
            joints[5 + i] = home
        currents = [0.0] * 15
        for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
            currents[i] = 4.5
        walking = abs(self.applied[0]) > 0.05
        gyro = (
            [
                0.7 * math.sin(2 * math.pi * 2.4 * self.t),
                0.6 * math.cos(2 * math.pi * 2.4 * self.t),
                0.0,
            ]
            if walking
            else [0.0, 0.0, 0.0]
        )  # a walking trunk rocks: a quiet trunk under a walk is a lifted one
        return RobotState.from_wire(
            {
                "t_ns": 10**10 + int(round(self.t * 1e9)),
                "policy": "walk" if walking else "stand",
                "move": {
                    "requested": [self.applied[0], 0, self.applied[1]],
                    "applied": [self.applied[0], 0, self.applied[1]],
                },
                "safety": {"fallen": False, "limp": False, "gravity": [0.0, 0.0, -1.0]},
                "joints": joints,
                "currents_ma": currents,
                "imu": {"gyro": gyro, "quat": [1.0, 0.0, 0.0, 0.0]},
                "odom": {"position": [self.odom[0], self.odom[1], 0.0], "yaw": self.odom[2]},
            }
        )

    def step(self) -> Row:
        state = self._state()
        left, ahead, right = self._tof()
        touch_tof = ahead is not None and ahead < 45
        depth = P.DepthView(
            t_ns=state.t_ns,
            ahead_mm=ahead,
            left_mm=left,
            right_mm=right,
            drop_zones=0,
            touch=touch_tof,
            head_pitch_mrad=0,
        )
        gap = self.nose_gap()
        # a gentle touch (the world's contact with a thing) is felt at its onset and fades
        in_contact = gap < 0.012 and (ahead is None or ahead > 60)
        if in_contact:
            self._contact_since = self.t if self._contact_since is None else self._contact_since
        else:
            self._contact_since = None
        felt = in_contact and self._contact_since is not None and self.t - self._contact_since < 1.2
        touch = C.TouchSense(site="body", pressure=0.2, source="model") if felt else C.TouchSense()
        walking = abs(self.applied[0]) > 0.05
        frame = C.SensorFrame(
            t_ms=int(round(self.t * 1000)),
            body=C.BodySense(
                posture="walk" if walking else "stand",
                vx=self.applied[0],
                vyaw=self.applied[1],
                odom=self.odom,
            ),
            tof=C.TofSense(
                present=True,
                left_mm=left,
                ahead_mm=ahead,
                right_mm=right,
                touch=touch_tof,
                age_ms=0,
            ),
            touch=touch,
            events=_events_of(self.last),
            sources={"body": "model"},
        )
        cmd, _tel = self.brain.step(frame.t_ms, frame)
        request = BodyRequest(
            vx=cmd.vx,
            vyaw=cmd.vyaw,
            head=None,
            posture="stand",
            approach=bool(cmd.approach),
            docking=bool(cmd.docking),
            back_arc=bool(cmd.back_arc),
        )
        out = self.link.tick(state, request, depth=depth)
        self.last = out
        vx, vyaw = out.twist
        self.applied = (vx, vyaw)
        if vx != 0.0:
            speed = (0.31 if vx > 0.0 else 0.25) * vx
            nyaw = self.yaw + (0.6 * vyaw - (0.05 if vx > 0.0 else 0.0)) * DT
            nx = self.x + speed * math.cos(nyaw) * DT
            ny = self.y + speed * math.sin(nyaw) * DT
            if self._ok(nx, ny, nyaw):
                self.x, self.y, self.yaw = nx, ny, nyaw
            else:  # slide along it; the yaw still turns where it can
                for tx, ty in ((nx, self.y), (self.x, ny)):
                    if self._ok(tx, ty, self.yaw):
                        self.x, self.y = tx, ty
                        break
                if self._ok(self.x, self.y, nyaw):
                    self.yaw = nyaw
        c0, s0 = math.cos(self.yaw0), math.sin(self.yaw0)
        dx, dy = self.x - self.x0, self.y - self.y0
        self.odom = (dx * c0 + dy * s0, -dx * s0 + dy * c0, self.yaw - self.yaw0)
        row = Row(
            t=round(self.t, 2),
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            vx=vx,
            vyaw=vyaw,
            want_vx=float(cmd.vx),
            want_vyaw=float(cmd.vyaw),
            tof_ahead=ahead,
            behaviour=str(cmd.behaviour),
            reflex=None if cmd.reflex is None else str(cmd.reflex),
            gate=tuple(out.gate),
            gap=gap,
            lean_in=cmd.reflex == "lean_in",
        )
        self.rows.append(row)
        self.t += DT
        return row

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / DT))):
            self.step()

    # ── the numbers the tests read ──────────────────────────────────────────

    def pushing_s(self) -> float:
        """Seconds the body link let a forward walk through while the nose touched something."""
        return sum(DT for r in self.rows if r.gap < 0.015 and r.vx > 0.05)

    def wedged_s(self) -> float:
        """Seconds the brain asked to walk forward with the nose against something (whatever the
        body link did with it)."""
        return sum(DT for r in self.rows if r.gap < 0.015 and r.want_vx > 0.05)

    def lean_in_s(self) -> float:
        return sum(DT for r in self.rows if r.lean_in)


# the bold explorer of tests/test_neuro_brain.py (growth day 120)
_BOLD_EXPLORER = dict(
    boldness=0.9,
    activity=0.9,
    sociability=0.3,
    exploration=0.9,
    wit=0.7,
    chronotype=0.3,
    tempo=0.6,
)


def make_brain(
    side: int = 1, seed: int = 1, behaviour: str = "wander", strength: float = 2.0
) -> NeuroBrain:
    """A bold explorer with its innate side ``side`` (+1 left, −1 right), the walk behaviour
    ``behaviour`` pushed on its own channel (an outside candidate: it competes like any other
    reason, the reflexes still go first). The base brain, no skills: this installs nothing for the
    rest of the test process (``install_default_skills`` is permanent — see
    :func:`make_integrated_brain`)."""
    genome = Genome.from_axes(_BOLD_EXPLORER, primary="explorer", branch="explorer", gd=120)
    genome = dataclasses.replace(genome, side=side, seed=seed)
    brain = NeuroBrain(genome, seed)
    brain.sel.set_extra({behaviour: strength})
    return brain


def make_integrated_brain(
    side: int = 1, seed: int = 1, behaviour: str = "wander", strength: float = 2.0
) -> NeuroBrain:
    """The production brain: the base brain with the play, social and sound skills.
    ``install_default_skills`` is permanent for the process it runs in — call this only in a
    subprocess (tests/test_neuro_unstuck.py ``_in_subprocess``), never in the pytest process: the
    behaviour tables other tests read would stay changed."""
    from duckmind.neuro.integration import install_default_skills
    from duckmind.neuro.integration import make_brain as integrated_brain
    from duckmind.verify.temperaments import genome as named_genome

    install_default_skills()
    genome = dataclasses.replace(named_genome("bold_explorer", seed), side=side)
    brain = integrated_brain(genome, seed)
    brain.sel.set_extra({behaviour: strength})
    return brain
