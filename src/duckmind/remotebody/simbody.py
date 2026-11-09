"""A simulated body on its own, behind the bridge: the rehearsal for a real Microduck.

``python -m duckmind.remotebody sim-body --token-file T --listen 127.0.0.1:8797 [--home H]`` starts
the MuJoCo body, the official ``robotd --sim`` (and the world's tofd and camera when it has a home)
exactly as the simulator page does, brings the duck up (enabled, seated), then hands it to whichever
brain dials in with the token — ``python -m duckmind.simweb ... --body remote://host:8797``. Its own
body object stays passive (it never writes to robotd); the bridge's safety keeper is the only other
writer, and only while no brain holds the link.

When the body is up it prints one line ``remote-body: ready {...}`` with its ports and sockets (the
rehearsal's scorer reads the world's truth through them, never the brain).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from duckmind.remotebody import link as L
from duckmind.remotebody.bridge import (
    Bridge,
    Channel,
    add_bridge_arguments,
    event_logger,
    host_port,
    keeper_for,
)


def _paths(args: argparse.Namespace) -> Any:
    from duckmind.livebody import LivePaths
    from duckmind.sim_live import _installed_ort_library

    if args.microduck:
        ort = args.ort_lib or _installed_ort_library()
        paths = LivePaths(args.microduck, args.microduck_rl, args.policies, args.robotd_bin, ort)
    else:
        paths = LivePaths.from_env()
    if paths is None:
        raise SystemExit("sim-body needs --microduck, --microduck-rl, --policies and --robotd-bin")
    missing = paths.missing()
    if missing:
        raise SystemExit("sim-body is missing: " + ", ".join(missing))
    return paths


def neural_duck_body(environ: MutableMapping[str, str]) -> dict[str, Any]:
    """Set up this process's environment for the body the neural duck's page would start for itself
    (``python -m duckmind.simweb --brain neuro``): the trained-motion table and servo delay, and the
    world server's neural-duck side (the truth its camera, hearing and touch models read: without
    it the remote brain runs blind and deaf, and its head never settles for a walk). A setting
    already in the environment wins, as on the page."""
    from duckmind.simweb.neuro_live import world_extensions
    from duckmind.skilltrain import motion_table

    installed = motion_table.install_defaults(environ)
    environ.setdefault("DUCKMIND_SIMHOME_EXT", world_extensions())
    return {**installed, "world_extensions": environ["DUCKMIND_SIMHOME_EXT"]}


def channels_for(body: Any) -> dict[str, Channel]:
    """What this simulated body offers a brain: robotd (its ``robot.health`` carrying the world's
    pack, as a real duck's robotd carries its own: :mod:`duckmind.remotebody.battery`), the world
    server and, when the world runs them, tofd and the camera's frame socket."""
    from duckmind.remotebody.battery import world_health_battery

    channels = {
        "robotd": Channel(
            "robotd", unix=str(body.robotd_socket), answers=world_health_battery(body.world)
        ),
        "world": Channel("world", tcp=("127.0.0.1", int(body.body_port))),
    }
    if body.home:
        sensors = body.world_call("x_sensors")
        tofd = (sensors.get("tofd") or {}).get("socket")
        media = (sensors.get("media") or {}).get("socket")
        if tofd:
            channels["tofd"] = Channel("tofd", unix=str(tofd))
        if media:
            channels["media"] = Channel("media", unix=str(media))
    return channels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckmind.remotebody sim-body",
        description="A simulated Microduck body served to a brain on the LAN (the rehearsal).",
    )
    add_bridge_arguments(parser)
    parser.add_argument("--home", help="the home its world runs (default: the page's default)")
    parser.add_argument("--no-nest", action="store_true", help="a duck without a nest")
    parser.add_argument("--tof", choices=("on", "off"), help="the ToF module fitted or not")
    parser.add_argument("--microduck", type=Path)
    parser.add_argument("--microduck-rl", type=Path)
    parser.add_argument("--policies", type=Path)
    parser.add_argument("--robotd-bin", type=Path)
    parser.add_argument("--ort-lib", type=Path)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="the plain official body: no trained-motion table, no servo delay and no neural "
        "duck world side (the neural duck's page sets all three up for its own body; so does "
        "this, unless --plain)",
    )
    args = parser.parse_args(argv)
    from duckmind.livebody import LiveBody

    log = event_logger(args.events)
    if not args.plain:
        log("neural_body", **neural_duck_body(os.environ))
    token = L.read_token(args.token_file)
    body = LiveBody(
        _paths(args),
        seated=True,
        home=args.home,
        nest=False if args.no_nest else None,
        tof=None if args.tof is None else args.tof == "on",
    )
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    bridge: Bridge | None = None
    try:
        body.start()
        body.passive = True  # one writer: the brain (or the bridge's safety keeper)
        channels = channels_for(body)
        bridge = Bridge(
            listen=host_port(args.listen),
            token=token,
            channels=channels,
            keeper=keeper_for(str(body.robotd_socket), "simulator", sit=not args.no_sit, log=log),
            body_info={
                "backend": "simulator",
                "home": body.home,
                "collision": body.collision,
                "lockstep": body.lockstep,
                "pose_joints": body.pose_joints_ok,
                "tof": body.tof,
                "head_alpha": body.robotd_head_alpha,
                "policies": body.policy_hashes(),
            },
            deadman_s=args.deadman_ms / 1000.0,
            log=log,
        )
        bridge.start()
        ready = {
            "port": bridge.port,
            "world_port": body.body_port,
            "robotd": str(body.robotd_socket),
            "channels": sorted(channels),
            "home": body.home,
        }
        print("remote-body: ready " + json.dumps(ready), flush=True)
        while not done.wait(1.0):
            if body.error:
                log("body_error", error=body.error)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if bridge is not None:
            bridge.close()
        body.stop()


if __name__ == "__main__":
    sys.exit(main())
