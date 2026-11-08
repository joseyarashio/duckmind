"""Pollen's anonymous usage data ("telemetry") on a real duck: read it, and turn it off.

Since daemon 0.18.3 a Microduck sends usage events (a random robot id, board, versions, policy
names) to PostHog, on by default (upstream ``docs/robot/telemetry.md``). **updaterd** sends them
and answers the switch: ``telemetry.status`` and ``telemetry.set {"enabled": false}`` on its own
socket ``/run/updaterd.sock`` (``duck-ipc-proto`` routes both to ``Service::Updater``; robotd
answers "… is not served by robotd"). The socket is ``0660 root:robot`` and neither call is in
``Call::is_mutating``, so the bridge's user — in the ``robot`` group for robotd's socket — may make
both. Off applies at once, survives reboots and updates, drops what is queued and deletes the
robot id; ``active`` is false whenever ``enabled`` is, or ``MICRODUCK_TELEMETRY=0`` /
``DO_NOT_TRACK=1`` is in updaterd's environment.

The owner's choice (2026-11-08) is off: the bridge on a real duck turns it off once when it starts
(:func:`settle`) unless told to keep it, and reports what it found to the brain's ``check``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

UPDATERD_SOCKET = "/run/updaterd.sock"


def _status(result: dict[str, Any]) -> dict[str, bool]:
    enabled, active = result.get("enabled"), result.get("active")
    if not isinstance(enabled, bool) or not isinstance(active, bool):
        raise ValueError(f"telemetry answer without enabled/active booleans: {result}")
    return {"enabled": enabled, "active": active}


def settle(
    socket_path: str | Path = UPDATERD_SOCKET,
    *,
    turn_off: bool = True,
    log: Callable[..., None] | None = None,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    """Ask updaterd for the telemetry switch and, with ``turn_off``, set it off when it is on.

    Returns what ``check`` prints: ``enabled`` / ``active`` as updaterd answers after this call,
    ``turned_off`` (this call switched it off), ``kept`` (asked not to touch it) and ``error``
    (updaterd not reachable, too old to know telemetry, or refused; the other fields are None
    then). Never raises: a duck whose switch cannot be read still runs."""
    from duckmind.robotd import RobotdProtocolError, _RpcConnection

    report: dict[str, Any] = {
        "socket": str(socket_path),
        "enabled": None,
        "active": None,
        "turned_off": False,
        "kept": not turn_off,
        "error": None,
    }
    try:
        rpc = _RpcConnection(socket_path, timeout_s)
    except OSError as error:
        report["error"] = f"cannot reach updaterd: {error}"
    else:
        try:
            now = _status(rpc.call("telemetry.status"))
            if turn_off and now["enabled"]:
                now = _status(rpc.call("telemetry.set", {"enabled": False}))
                report["turned_off"] = True
            report.update(now)
        except (OSError, RobotdProtocolError, ValueError) as error:
            report["error"] = str(error)
        finally:
            rpc.close()
    if log is not None:
        log("telemetry", **report)
    return report


def describe(report: object) -> str:
    """One plain line for ``check`` from the bridge's :func:`settle` report."""
    if not isinstance(report, dict):
        return "usage data (Pollen telemetry): not reported by this bridge"
    if report.get("error"):
        return (
            f"usage data (Pollen telemetry): UNKNOWN, updaterd did not answer "
            f"({report['error']}); turn it off on the duck: the console's 'share anonymous usage "
            'data\' switch, or telemetry.set {"enabled": false} on /run/updaterd.sock'
        )
    state = f"enabled {report.get('enabled')}, active {report.get('active')}"
    if report.get("active"):
        how = (
            "the bridge was told to keep it (--keep-pollen-telemetry)"
            if report.get("kept")
            else "updaterd did not take the switch"
        )
        return f"usage data (Pollen telemetry): ON, the duck sends it ({state}); {how}"
    if report.get("turned_off"):
        done = "the bridge turned it off at start"
    elif report.get("enabled"):
        done = "updaterd's environment blocks it: MICRODUCK_TELEMETRY=0 or DO_NOT_TRACK=1"
    else:
        done = "already off"
    return f"usage data (Pollen telemetry): off ({state}; {done})"
