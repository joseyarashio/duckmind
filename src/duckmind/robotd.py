"""Typed, deliberately narrow client for official robotd JSON-RPC.

The operator's enable control is separate from the BodyPort handed to skills.
Neither client exposes servo, gain, torque, policy-load or reboot methods.
"""

from __future__ import annotations

import json
import math
import select
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from duckmind.body import BodyAck, BodyIntent, RobotObservation, SafetyObservation

API_VERSION = 37  # A liveness hint, not a version gate in current robotd.
MAX_FRAME_BYTES = 2_000_000
# Current v5 velstand policy observes head commands as deltas from this HOME.
# robot.look currently returns absolute targets and must not be used as the
# first-board gameplay primitive until that upstream contract is reconciled.
V5_HOME_HEAD = (0.3491, 0.3491, 0.0, 0.0)


class RobotdProtocolError(RuntimeError):
    """The daemon's response is missing, malformed or a JSON-RPC error."""


class _JsonLineSocket:
    def __init__(self, path: str | Path, timeout_s: float) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout_s)
        self._socket.connect(str(path))
        self._buffer = bytearray()

    def close(self) -> None:
        self._socket.close()

    def send(self, payload: Mapping[str, object]) -> None:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_FRAME_BYTES:
            raise RobotdProtocolError("request frame too large")
        self._socket.sendall(encoded)

    def read(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            line_end = self._buffer.find(b"\n")
            if line_end >= 0:
                line = bytes(self._buffer[:line_end])
                del self._buffer[: line_end + 1]
                try:
                    result = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RobotdProtocolError("invalid JSON frame") from error
                if not isinstance(result, dict):
                    raise RobotdProtocolError("JSON-RPC frame must be an object")
                return result
            if len(self._buffer) >= MAX_FRAME_BYTES:
                raise RobotdProtocolError("response frame too large")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._socket], [], [], remaining)[0]:
                raise TimeoutError("robotd response/state timeout")
            data = self._socket.recv(65536)
            if not data:
                raise ConnectionError("robotd closed the socket")
            self._buffer.extend(data)


class _RpcConnection:
    def __init__(self, path: str | Path, timeout_s: float) -> None:
        self.wire = _JsonLineSocket(path, timeout_s)
        self.timeout_s = timeout_s
        self._next_id = 1
        self._lock = (
            threading.Lock()
        )  # (the operator is shared by the driver thread and the body's stop)

    def close(self) -> None:
        self.wire.close()

    def call(self, method: str, params: Mapping[str, object] | None = None) -> dict[str, Any]:
        with self._lock:
            return self._call(method, params)

    def _call(self, method: str, params: Mapping[str, object] | None = None) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self.wire.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params or {}),
            }
        )
        # A call that timed out leaves its answer on the way: it arrives later under an older id
        # (a request the daemon could not parse is answered with an error and no id). That is not
        # this call's answer: skip it and keep reading, so one stall never leaves the link a step
        # behind.
        deadline = time.monotonic() + self.timeout_s
        while True:
            response = self.wire.read(max(0.0, deadline - time.monotonic()))
            answered = response.get("id")
            stray = (
                isinstance(answered, int)
                and not isinstance(answered, bool)
                and answered < request_id
            ) or (answered is None and "error" in response)
            if not stray:
                break
        if answered != request_id:
            raise RobotdProtocolError(f"unexpected response id for {method}")
        if "error" in response:
            raise RobotdProtocolError(f"{method}: {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RobotdProtocolError(f"{method}: missing result object")
        return result

    def notify(self, method: str, params: Mapping[str, object]) -> None:
        self.wire.send({"jsonrpc": "2.0", "method": method, "params": dict(params)})


def _ack(result: Mapping[str, object]) -> BodyAck:
    accepted = result.get("accepted")
    if not isinstance(accepted, bool):
        raise RobotdProtocolError("intent response has no accepted boolean")
    reason = result.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise RobotdProtocolError("intent reason is not text")
    return BodyAck(accepted, reason)


def _vector3(value: object, field: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in value
        )
    ):
        raise RobotdProtocolError(f"invalid {field} vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _head_values(
    values: Mapping[str, object], *, missing_zero: bool = False
) -> tuple[float, float, float, float]:
    names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    parsed: list[float] = []
    for name in names:
        value = values.get(name, 0.0 if missing_zero else None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RobotdProtocolError(f"invalid {name} head target")
        parsed.append(float(value))
    return (parsed[0], parsed[1], parsed[2], parsed[3])


def _observation(frame: Mapping[str, object], backend: str) -> RobotObservation:
    if frame.get("method") != "robot.state":
        raise RobotdProtocolError("expected robot.state notification")
    params = frame.get("params")
    if not isinstance(params, dict):
        raise RobotdProtocolError("robot.state has no params object")
    movement = params.get("move")
    safety = params.get("safety")
    if not isinstance(movement, dict) or not isinstance(safety, dict):
        raise RobotdProtocolError("robot.state lacks movement or safety")
    policy = params.get("policy")
    robot_time = params.get("t")
    limited = movement.get("limited_by", [])
    if (
        not isinstance(policy, str)
        or not isinstance(robot_time, (int, float))
        or not math.isfinite(robot_time)
    ):
        raise RobotdProtocolError("robot.state lacks policy or daemon time")
    if not isinstance(limited, list) or any(not isinstance(item, str) for item in limited):
        raise RobotdProtocolError("invalid limited_by")
    if not isinstance(safety.get("fallen"), bool) or not isinstance(safety.get("limp"), bool):
        raise RobotdProtocolError("invalid safety flags")
    # Only hardware-observable information crosses this port. Joints/targets are
    # intentionally omitted; simulator-only truth is a separate evaluator port.
    metadata = {"odom": params["odom"]} if isinstance(params.get("odom"), dict) else {}
    return RobotObservation(
        monotonic_s=time.monotonic(),
        policy=policy,
        movement_requested=_vector3(movement.get("requested"), "requested"),
        movement_applied=_vector3(movement.get("applied"), "applied"),
        limited_by=tuple(limited),
        safety=SafetyObservation(fallen=safety["fallen"], limp=safety["limp"]),
        backend=backend,
        metadata=metadata,
        robot_time_s=float(robot_time),
    )


class RobotdBodyPort:
    """Same skill-facing IPC on a real robot and on `robotd --sim`."""

    def __init__(
        self, path: str | Path, *, backend: str, timeout_s: float = 2.0, hz: int = 25
    ) -> None:
        if backend not in {"real", "simulator"}:
            raise ValueError("backend must be real or simulator")
        if timeout_s <= 0 or hz <= 0 or hz > 50:
            raise ValueError("invalid socket timeout or state rate")
        self._command = _RpcConnection(path, timeout_s)
        try:
            self.hello = self._command.call("hello", {"api_version": API_VERSION})
            self._state = _RpcConnection(path, timeout_s)
            try:
                self.capabilities = self._state.call("robot.subscribe", {"hz": hz})
                if self.capabilities.get("accepted") is not True:
                    raise RobotdProtocolError("robot.subscribe was refused")
            except BaseException:
                self._state.close()
                raise
        except BaseException:
            self._command.close()
            raise
        self.backend = backend
        self._look_target: tuple[float, float, float, float] | None = None

    def close(self) -> None:
        self._state.close()
        self._command.close()

    def __enter__(self) -> RobotdBodyPort:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def supports_v5_head_profile(self) -> bool:
        return (
            self.hello.get("api_version") == API_VERSION
            and self.capabilities.get("walk") == "velstand.onnx"
        )

    def dispatch_intent(self, intent: BodyIntent) -> BodyAck:
        if intent.method in {"robot.move", "robot.head", "robot.mouth"}:
            if intent.method == "robot.head" and not self.supports_v5_head_profile:
                raise RobotdProtocolError("v5 head profile is not verified for this daemon/policy")
            self._command.notify(intent.method, intent.params)
            if intent.method == "robot.head":
                delta = _head_values(intent.params, missing_zero=True)
                self._look_target = (
                    V5_HOME_HEAD[0] + delta[0],
                    V5_HOME_HEAD[1] + delta[1],
                    V5_HOME_HEAD[2] + delta[2],
                    V5_HOME_HEAD[3] + delta[3],
                )
            return BodyAck(True, "sent_unconfirmed", confirmed=False)
        result = self._command.call(intent.method, intent.params)
        if intent.method == "robot.look":
            head = result.get("head")
            if not isinstance(head, dict) or not isinstance(result.get("clamped"), bool):
                raise RobotdProtocolError("robot.look lacks head target or clamp flag")
            self._look_target = _head_values(head)
            return BodyAck(True, "clamped" if result["clamped"] else None)
        return _ack(result)

    def observe(self) -> RobotObservation:
        frame = self._state.wire.read(self._state.timeout_s)
        observation = _observation(frame, self.backend)
        if self._look_target is None:
            return observation
        params = frame["params"]
        joints = params.get("joints")
        if (
            not isinstance(joints, list)
            or len(joints) != 15
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in joints[5:9]
            )
        ):
            return observation  # missing measurement is unknown, never success
        error = max(
            abs(float(joints[index + 5]) - target) for index, target in enumerate(self._look_target)
        )
        return replace(observation, head_tracking_error_rad=error)

    def stop(self) -> BodyAck:
        return _ack(self._command.call("robot.stop"))


class RobotdOperatorClient:
    """Separate commissioning authority, never handed to cognition or skills."""

    def __init__(self, path: str | Path, *, timeout_s: float = 2.0) -> None:
        self._rpc = _RpcConnection(path, timeout_s)
        try:
            self._rpc.call("hello", {"api_version": API_VERSION})
        except BaseException:
            self._rpc.close()
            raise

    def close(self) -> None:
        self._rpc.close()

    def __enter__(self) -> RobotdOperatorClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def enable(self, on: bool) -> BodyAck:
        if not isinstance(on, bool):
            raise ValueError("enable requires a boolean")
        return _ack(self._rpc.call("robot.enable", {"on": on}))

    def init(self) -> BodyAck:
        """``robot.init``: torque back on every servo (some may hang free after a scripted pose), then the
        two-second ramp to the home pose. The policy is brought back with :meth:`enable`."""
        return _ack(self._rpc.call("robot.init", {}))

    def pose_joints(
        self,
        targets: Sequence[float | None],
        off: Sequence[str],
        *,
        gain: int = 160,
        ramp_s: float = 1.0,
    ) -> BodyAck:
        """``robot.poseJoints`` (microduck_emotions' workshop patch, the simulator's robotd build only): a scripted
        joint pose. ``targets``: radians in robotd's joint order (``None``: hold where it is), ``off``: joints whose
        torque is cut (they hang free), the rest ramp to their targets over ``ramp_s`` at servo gain ``gain`` and
        hold, with the policy off, until :meth:`init`. A robotd without the method answers with an error
        (:class:`RobotdProtocolError`)."""
        if len(targets) > 15 or any(
            t is not None and (isinstance(t, bool) or not math.isfinite(float(t))) for t in targets
        ):
            raise ValueError("poseJoints wants up to 15 finite targets")
        if not 0.0 <= ramp_s <= 10.0 or not 0 < int(gain) <= 400:
            raise ValueError("poseJoints: ramp_s 0..10 s, gain 1..400")
        return _ack(
            self._rpc.call(
                "robot.poseJoints",
                {
                    "targets": [None if t is None else float(t) for t in targets],
                    "off": [str(n) for n in off],
                    "gain": int(gain),
                    "ramp_s": float(ramp_s),
                },
            )
        )


class RobotdSafetyClient:
    """Independent stop/health socket so a blocked skill stream cannot delay stop."""

    def __init__(self, path: str | Path, *, timeout_s: float = 1.0) -> None:
        self._rpc = _RpcConnection(path, timeout_s)
        try:
            self._rpc.call("hello", {"api_version": API_VERSION})
        except BaseException:
            self._rpc.close()
            raise

    def close(self) -> None:
        self._rpc.close()

    def __enter__(self) -> RobotdSafetyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def stop(self) -> BodyAck:
        return _ack(self._rpc.call("robot.stop"))

    def health(self) -> dict[str, object]:
        result = self._rpc.call("robot.health")
        if not isinstance(result.get("healthy"), bool):
            raise RobotdProtocolError("robot.health lacks healthy verdict")
        battery = result.get("battery")
        if battery is not None:
            if not isinstance(battery, dict):
                raise RobotdProtocolError("robot.health battery is malformed")
            percent = battery.get("percent")
            if (
                isinstance(percent, bool)
                or not isinstance(percent, (int, float))
                or not math.isfinite(percent)
                or not 0 <= percent <= 100
            ):
                raise RobotdProtocolError("robot.health battery percent is invalid")
        return result

    def policies(self) -> dict[str, object]:
        """Read robotd's loaded slot report; never changes a policy."""

        return self._rpc.call("robot.policies")
