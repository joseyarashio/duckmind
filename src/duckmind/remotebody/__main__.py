"""``python -m duckmind.remotebody <command>``: token, bridge, sim-body, shim."""

from __future__ import annotations

import sys
from pathlib import Path

from duckmind.remotebody.doors import BOARD_DOC, CLOSE_SCRIPT, MEDIAD_PORTS, open_ports

USAGE = """usage: python -m duckmind.remotebody <command> [options]

  token PATH   write a fresh shared token to PATH (mode 0600); copy the same file to the other machine
  check URL    on the computer: dial remote://HOST:PORT, report the link and robotd's health (moves nothing)
  bridge       on the duck: serve robotd (and tofd, mediad) to one brain on the home LAN
  sim-body     a simulated duck behind the bridge (rehearse without the robot)
  shim         a delay / jitter / loss / cut shim in front of a bridge (testing on one machine)
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    command, rest = args[0], args[1:]
    if command == "token":
        if len(rest) != 1:
            print("usage: python -m duckmind.remotebody token PATH", file=sys.stderr)
            return 2
        from duckmind.remotebody.link import new_token

        new_token(Path(rest[0]).expanduser())
        print(f"wrote a new token to {rest[0]} (keep it private; copy it to the other machine)")
        return 0
    if command == "check":
        return check(rest)
    if command == "bridge":
        from duckmind.remotebody.bridge import main as run
    elif command == "sim-body":
        from duckmind.remotebody.simbody import main as run
    elif command == "shim":
        from duckmind.remotebody.netshim import main as run
    else:
        print(USAGE, file=sys.stderr)
        return 2
    return run(rest)


def check(argv: list[str]) -> int:
    """Dial the body, hold the link for a few seconds, ask robotd who it is and how it is (hello,
    robot.health, robot.policies: read-only), and print the link's round trip, the clock offset,
    the battery, the camera's frame rate over the link, and whether mediad's LAN doors answer.
    ``--wifi-minutes M`` then holds the link M minutes and reports the home Wi-Fi's quality
    (:mod:`duckmind.remotebody.quality`). Moves nothing. The last line is the verdict: ``ready``
    (exit 0), or ``NOT READY`` with why (exit 1) — the link dropped, or one of Pollen's LAN doors
    is open (:mod:`duckmind.remotebody.doors`): a LAN peer could drive the duck around the bridge."""
    import argparse
    import json
    import time

    from duckmind import paths
    from duckmind.remotebody.battery import parse
    from duckmind.remotebody.client import RemoteLink
    from duckmind.remotebody.link import read_token
    from duckmind.robotd import RobotdSafetyClient

    parser = argparse.ArgumentParser(prog="python -m duckmind.remotebody check")
    parser.add_argument("url", help="remote://HOST:PORT")
    parser.add_argument(
        "--token-file", type=Path, default=paths.duckmind_home() / "remote-body.token"
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--camera-seconds",
        type=float,
        default=3.0,
        help="how long to fetch camera frames over the link (0: skip)",
    )
    parser.add_argument(
        "--wifi-minutes",
        type=float,
        default=0.0,
        help="then hold the link this long and report the Wi-Fi's quality (10 for the checklist)",
    )
    parser.add_argument(
        "--mediad-ports",
        default=",".join(str(p) for p in MEDIAD_PORTS),
        help="mediad's LAN ports to knock on (empty: skip)",
    )
    args = parser.parse_args(argv)
    link = RemoteLink(args.url, read_token(args.token_file))
    try:
        link.start(timeout_s=10)
        print(f"link up: channels {', '.join(link.channels)}; body {json.dumps(link.info)}")
        for line in describe_body(link.info):
            print(line)
        print(f"the body stops and sits after {link.deadman_ms} ms without this computer")
        with RobotdSafetyClient(link.robotd_socket, timeout_s=2.0) as robotd:
            health = robotd.health()
            print(f"robot.health: {json.dumps(health)}")
            print(describe_battery(parse(health)))
            print(f"robot.policies: {json.dumps(robotd.policies())[:400]}")
        time.sleep(args.seconds)
        stats = link.stats()
        print(
            f"round trip p50 {stats['rtt_ms_p50']} ms, p95 {stats['rtt_ms_p95']} ms, "
            f"longest silence {stats['max_gap_ms']} ms, drops {stats['drops']}"
        )
        print(describe_clock(link.quality))
        media = link.endpoint("media")
        if media is None:
            print("camera over the link: the bridge serves no camera (start it with --media)")
        elif args.camera_seconds > 0:
            print(describe_camera(camera_rate(media, args.camera_seconds)))
        not_ready = []
        ports = [int(p) for p in str(args.mediad_ports).split(",") if p.strip()]
        reachable = open_ports(link.url.host, ports) if ports else []
        if ports:
            for line in describe_doors(link.url.host, reachable):
                print(line)
        if reachable:
            not_ready.append("mediad's LAN doors answer from this computer")
        found = link.info.get("doors")
        if isinstance(found, dict):
            for line in describe_bridge_doors(found):
                print(line)
            if found.get("netpadd") is not None:
                not_ready.append("netpadd listened on the duck when the bridge started")
        drops = int(stats["drops"])
        if args.wifi_minutes > 0:
            drops = wifi_quality(link, args.wifi_minutes * 60.0)["drops"]
        if drops:
            not_ready.append(f"the link dropped {drops} time(s)")
        if not_ready:
            print("NOT READY: " + "; ".join(not_ready))
            return 1
        knocked = "no door round the bridge answered" if ports else "mediad's doors not knocked on"
        print(f"ready: the link held, and {knocked}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"check failed: {error}", file=sys.stderr)
        return 1
    finally:
        link.close()


def describe_battery(battery: dict[str, object] | None) -> str:
    """robotd's battery as the brain gets it on a remote body (:mod:`duckmind.remotebody.battery`)."""
    if battery is None:
        return "battery: robotd has no reading yet (unknown, not empty)"
    charging = battery.get("charging")
    said = "not reported by robotd" if charging is None else ("yes" if charging else "no")
    return (
        f"battery (what the brain gets): {battery['percent']:.0f} % ({battery.get('volts')} V, "
        f"the servos' supply; it sags while walking), charging: {said}"
    )


def describe_clock(quality: object) -> str:
    offset = getattr(quality, "wall_offset_ms", None)
    if offset is None:
        return "clock: the bridge did not send its wall clock (an older bridge)"
    error = getattr(quality, "wall_offset_err_ms", None) or 0.0
    return (
        f"clock: the duck's wall clock is {offset:+.0f} ms from this computer's (± {error:.0f} ms); "
        "add it to this computer's times to line them up with the bridge's events file (the "
        "brain itself uses robotd's own clock, no sync needed)"
    )


def camera_rate(
    socket_path: Path, seconds: float, fetch: object = None, clock: object = None
) -> dict[str, object]:
    """Fetch camera frames back to back over the link for ``seconds`` (one ``media.frame`` per
    connection, as duck-eye does): how many came, how fast, how big."""
    import time

    from duckmind.eye import frames

    get = fetch if callable(fetch) else frames.fetch
    now = clock if callable(clock) else time.monotonic
    begun = now()
    count, total, errors, size, last_error = 0, 0, 0, None, None
    while now() - begun < seconds:
        try:
            header, data = get(socket_path)
        except (OSError, RuntimeError) as error:
            errors += 1
            last_error = str(error)
            if errors >= 3 and count == 0:
                break
            continue
        count += 1
        total += len(data)
        size = (header.width, header.height)
    spent = max(now() - begun, 1e-9)
    return {
        "frames": count,
        "seconds": round(spent, 2),
        "fps": round(count / spent, 2),
        "size": size,
        "mbit_s": round(total * 8 / spent / 1e6, 1),
        "errors": errors,
        "error": last_error,
    }


def describe_camera(rate: dict[str, object]) -> str:
    if not rate["frames"]:
        return f"camera over the link: no frame in {rate['seconds']} s ({rate['error']})"
    size = rate["size"]
    assert isinstance(size, tuple)
    return (
        f"camera over the link: {rate['fps']} frames/s ({rate['frames']} in {rate['seconds']} s, "
        f"{size[0]}×{size[1]} UYVY, {rate['mbit_s']} Mbit/s, {rate['errors']} failed)"
    )


def describe_doors(host: str, reachable: list[int]) -> list[str]:
    if not reachable:
        return [f"mediad's LAN doors: closed from this computer (knocked on {host})"]
    ports = ", ".join(str(p) for p in reachable)
    return [
        f"mediad's LAN doors OPEN from this computer: {host} port {ports} (no password; anyone on "
        "this network can drive the duck through them, around the bridge)",
        f"  to close them, on the duck: sudo sh {CLOSE_SCRIPT} --brain <this computer's "
        f"address> ({BOARD_DOC})",
    ]


def describe_bridge_doors(found: dict[str, object]) -> list[str]:
    """What a real duck's bridge found at those doors, on the duck, when it started."""
    out = []
    if found.get("netpadd") is not None:
        out.append(
            f"netpadd's LAN door: something listened on UDP {found['netpadd']} on the duck when "
            "the bridge started (a gamepad on the network drives the duck around the bridge)"
        )
        out.append(
            "  turn it off on the duck: [netpad] enabled = false (sudo robotctl configure, "
            "then restart robotd and the bridge)"
        )
    if found.get("allowed") and (found.get("tcp_open") or found.get("netpadd") is not None):
        out.append(
            "the bridge was started with --allow-open-mediad while a door was open: its "
            "deadman does not cover that door"
        )
    return out


def wifi_quality(link: object, seconds: float, minute_s: float = 60.0) -> dict[str, object]:
    """Hold the link ``seconds`` and say, every ``minute_s`` and at the end, how the Wi-Fi did:
    round trips, the longest silence at the duck, drops — every drop while this computer kept
    pinging is a false deadman alarm (the duck sits down for nothing) — and near misses."""
    import time

    quality = getattr(link, "quality")
    deadman = getattr(link, "deadman_ms", None)
    print(f"Wi-Fi quality: holding the link for {seconds / 60:.1f} min (pings every 100 ms)")
    begun = time.monotonic()
    drops0 = int(getattr(link, "drops", 0))
    start = quality.mark()
    minute = 0
    while True:
        left = seconds - (time.monotonic() - begun)
        if left <= 0:
            break
        window, window_drops = quality.mark(), int(getattr(link, "drops", 0))
        time.sleep(min(minute_s, left))
        minute += 1
        part = quality.summary(
            seconds=min(minute_s, left),
            drops=int(getattr(link, "drops", 0)) - window_drops,
            deadman_ms=deadman,
            since=window,
        )
        print(f"  minute {minute}: {_quality_line(part)}")
    total = quality.summary(
        seconds=time.monotonic() - begun,
        drops=int(getattr(link, "drops", 0)) - drops0,
        deadman_ms=deadman,
        since=start,
    )
    print(f"Wi-Fi quality over {total['seconds']} s: {_quality_line(total)}")
    print(
        f"  false deadman alarms: {total['drops']} ({total['false_alarms_per_hour']} per hour at "
        f"the {deadman} ms deadman); near misses (a silence at the duck over half the deadman): "
        f"{total['near_misses']} ({total['near_misses_per_hour']} per hour)"
    )
    return total


def _quality_line(q: dict[str, object]) -> str:
    return (
        f"round trip p50 {q['rtt_ms_p50']} / p95 {q['rtt_ms_p95']} / max {q['rtt_ms_max']} ms, "
        f"longest silence at the duck {q['duck_silence_ms_max']} ms, drops {q['drops']}"
    )


def describe_body(info: dict[str, object]) -> list[str]:
    """What the bridge said about the duck (its walk and sit-stand networks, its robotd.toml, its
    camera, Pollen's usage data) in plain lines for ``check``: whether the brain has measured those
    networks in the simulator (it drives an unmeasured walk with the reference walk's numbers),
    whether the duck sends usage data, and what the owner should change on the duck."""
    from duckmind.eye.camera import BOARD_CAMERAS
    from duckmind.neuro import seatcal, walkcal

    out: list[str] = []
    policies = info.get("policies")
    if isinstance(policies, dict):
        walks, seats = walkcal.load(), seatcal.load()
        walk, seat = policies.get("walk_sha256"), policies.get("sitstand_sha256")
        known_walk = walks.walks.get(walk) if isinstance(walk, str) else None
        known_seat = seats.seats.get(seat) if isinstance(seat, str) else None
        out.append(f"policy set: {policies.get('set') or 'unknown'}")
        if walk is None:
            out.append("walk: not the official velstand set (the start checks refuse it)")
        elif known_walk is not None:
            out.append(f"walk: {known_walk.name}, sha256 {walk} (measured in the simulator)")
        else:
            out.append(
                f"walk: sha256 {walk}, NOT measured in the simulator: the brain drives it with "
                "the reference walk's numbers"
            )
        if known_seat is not None:
            out.append(f"sit-stand: {known_seat.name}, sha256 {seat} (measured in the simulator)")
        else:
            out.append(f"sit-stand: sha256 {seat}, NOT measured in the simulator")
    config = info.get("robotd_config")
    if isinstance(config, dict):
        layers = config.get("layers") or {}
        out.append(
            f"robotd.toml: board {config.get('board')}, idle {config.get('idle')}, pickup "
            f"{config.get('pickup')}, limp_fall {config.get('limp_fall')} (owner layer "
            f"{'present' if isinstance(layers, dict) and layers.get('user') else 'absent'})"
        )
        if config.get("idle"):
            out.append(
                "  change on the duck: [idle] enabled = false (sudo robotctl configure, then "
                "restart robotd): robotd's look-around and yawn take the head and open the beak"
            )
        family = BOARD_CAMERAS.get(str(config.get("board")))
        if family is None:
            out.append(f"camera: no lens known for board {config.get('board')}")
        else:
            out.append(
                f"camera: {family.sensor} ({family.name} family, {family.view_deg[0]}° × "
                f"{family.view_deg[1]}°): duck-eye --board {config.get('board')}"
            )
    elif info.get("backend") == "real":
        out.append("robotd.toml: not readable by the bridge's user (/etc/robot/robotd.toml)")
    if info.get("backend") == "real":
        from duckmind.remotebody.telemetry import describe

        out.append(describe(info.get("telemetry")))
    return out


if __name__ == "__main__":
    sys.exit(main())
