"""``python -m duckmind.remotebody <command>``: token, bridge, sim-body, shim."""

from __future__ import annotations

import sys
from pathlib import Path

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
    robot.health, robot.policies: read-only), and print the link's round trip. Moves nothing."""
    import argparse
    import json
    import time

    from duckmind import paths
    from duckmind.remotebody.client import RemoteLink
    from duckmind.remotebody.link import read_token
    from duckmind.robotd import RobotdSafetyClient

    parser = argparse.ArgumentParser(prog="python -m duckmind.remotebody check")
    parser.add_argument("url", help="remote://HOST:PORT")
    parser.add_argument(
        "--token-file", type=Path, default=paths.duckmind_home() / "remote-body.token"
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    link = RemoteLink(args.url, read_token(args.token_file))
    try:
        link.start(timeout_s=10)
        print(f"link up: channels {', '.join(link.channels)}; body {json.dumps(link.info)}")
        print(f"the body stops and sits after {link.deadman_ms} ms without this computer")
        with RobotdSafetyClient(link.robotd_socket, timeout_s=2.0) as robotd:
            print(f"robot.health: {json.dumps(robotd.health())}")
            print(f"robot.policies: {json.dumps(robotd.policies())[:400]}")
        time.sleep(args.seconds)
        stats = link.stats()
        print(
            f"round trip p50 {stats['rtt_ms_p50']} ms, p95 {stats['rtt_ms_p95']} ms, "
            f"longest silence {stats['max_gap_ms']} ms, drops {stats['drops']}"
        )
        return 0 if stats["drops"] == 0 else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"check failed: {error}", file=sys.stderr)
        return 1
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
