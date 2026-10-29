"""Best-effort independent robotd stop used by systemd after service exit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from duckmind.robotd import RobotdSafetyClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robotd", type=Path, default=Path("/run/robotd.sock"))
    args = parser.parse_args()
    try:
        with RobotdSafetyClient(args.robotd, timeout_s=1.0) as client:
            accepted = client.stop().accepted
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as error:
        print(f"Duck Mind post-stop could not reach robotd: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if not accepted:
        print("Duck Mind post-stop was refused by robotd", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
