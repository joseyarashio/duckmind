"""duck-body as a lockstep brain: the same :class:`~duckmind.duckbody.bodyloop.BodyLoop` the
process runs, fed by a rig's sensor doors instead of sockets (phase-3 architecture rule: the
headless rig and the P2 determinism tests drive the same code as ``server.py``).

:class:`BodyBrain` is duck-typed to the rig's ``Brain`` (``duckmind.world.rig``; duck-body never
imports the world): ``tick(doors)`` takes an object with ``t_ns``, ``state`` (a robot.state
params dict), ``tof`` (a tof.frame params dict or ``None``), ``results`` (eye.result lines) and
``health`` (robot.health or ``None``) and returns an object with ``intents`` (guarded
:class:`~duckmind.duckbody.protocol.Intent`), ``looks`` (orders with ``look`` = the ``eye.look``
line and ``poses`` = the posture tap's ``eye.pose`` samples, read when the rig captures the frame)
and ``percepts`` (the percept-v1 events sensed this tick). ``see`` hands a captured frame to
duck-eye's pure look function when one is given.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from duckmind.duckbody import protocol as P
from duckmind.duckbody.bodyloop import BodyLoop
from duckmind.duckbody.robotd_client import RobotState

EyeFn = Callable[[Mapping[str, Any], Mapping[str, Any], bytes, Sequence[Mapping[str, Any]]], dict]
POSE_WINDOW_NS = 2_000_000_000


class _LivePoses(Sequence[Mapping[str, Any]]):
    """The tap's recent ``eye.pose`` samples, read at access time (the capture comes later)."""

    def __init__(self, loop: BodyLoop) -> None:
        self.loop = loop

    def _now(self) -> list[Mapping[str, Any]]:
        ring = list(self.loop.tap.ring)
        if not ring:
            return []
        newest = ring[-1].t_ns
        return [s.to_wire() for s in ring if newest - s.t_ns <= POSE_WINDOW_NS and s.seq % 2 == 0]

    def __len__(self) -> int:
        return len(self._now())

    def __getitem__(self, index: Any) -> Any:
        return self._now()[index]

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._now())


@dataclass(frozen=True)
class Order:
    look: dict[str, Any]
    poses: Sequence[Mapping[str, Any]]


@dataclass
class Out:
    intents: list[P.Intent] = field(default_factory=list)
    looks: list[Order] = field(default_factory=list)
    percepts: list[Mapping[str, Any]] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)


class BodyBrain:
    def __init__(
        self,
        loop: BodyLoop | None = None,
        eye: EyeFn | None = None,
        standing_height_m: float = 0.12,
    ) -> None:
        self.loop = loop or BodyLoop()
        self.eye = eye
        self.standing_height_m = standing_height_m
        self.depth: P.DepthView | None = None
        self.tracker: Any = None
        self.poses = _LivePoses(self.loop)

    def _tof(self, frame: Mapping[str, Any]) -> None:
        from duckmind.duckbody import depth as D
        from duckmind.duckbody import tof_reproject as T

        ring = self.loop.tap.ring
        pose = ring[-1] if ring else None
        if pose is None or pose.tof_pos is None or pose.tof_quat is None:
            return
        if self.tracker is None:
            self.tracker = D.DepthTracker(standing_height_m=self.standing_height_m)
        state = self.loop.state
        down = state.down if state is not None else None
        reading, _ = self.tracker.step(
            D.ToFFrame.of(
                int(frame.get("t_ns", 0) or pose.t_ns),
                frame.get("distance_mm", []),
                frame.get("status", []),
            ),
            D.SensorPose(
                T.Pose(pose.tof_pos, T.Quat(*pose.tof_quat)),
                down or (0.0, 0.0, -1.0),
                pose.trunk_height_mm,
            ),
            walking=self.loop._during() == "walk",
        )
        self.depth = D.body_view(reading)

    def command(self, obj: Mapping[str, Any], t_ns: int) -> list[dict[str, Any]]:
        """A body.sock command, as the process would take it (acks back)."""
        return self.loop.command(obj, t_ns)

    def tick(self, doors: Any) -> Out:
        state = RobotState.from_wire(doors.state)
        if getattr(doors, "tof", None):
            self._tof(doors.tof)
        tick = self.loop.tick(
            state, depth=self.depth, eye=list(getattr(doors, "results", ()) or ())
        )
        out = Out(intents=list(tick.intents), lines=list(tick.lines))
        health = getattr(doors, "health", None)
        if health:
            out.lines += self.loop.health(health, state.t_ns)
        seq = tick.pose.seq if tick.pose is not None else 0
        for request in tick.looks:
            line = {
                "type": "eye.look",
                "look_id": request.look_id,
                "attempt": 0,
                "kind": getattr(request, "kind", "end_look"),
                "tasks": list(request.tasks),
                "pose_tap_ref": seq,
                "not_before_ns": request.not_before_ns,
                "deadline_ns": None,
                "media_id": 2,
                "mode": getattr(request, "mode", "awake"),
                "hints": dict(getattr(request, "hints", {}) or {}),
            }
            out.looks.append(Order(line, self.poses))
        out.percepts = [line["event"] for line in out.lines if line.get("type") == "percept"]
        return out

    def see(
        self,
        look: Mapping[str, Any],
        header: Mapping[str, Any],
        payload: bytes,
        poses: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if self.eye is None:
            return {
                "type": "eye.result",
                "look_id": look.get("look_id"),
                "status": "none",
                "why": "no_eye",
            }
        return self.eye(look, header, payload, list(poses))


__all__ = ["BodyBrain", "Order", "Out"]
