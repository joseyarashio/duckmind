"""Local operator CLI; it has no direct motor or database access."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import uuid
from pathlib import Path


def request(socket_path: Path, payload: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(8.0)
        client.connect(str(socket_path))
        client.sendall(encoded)
        with client.makefile("rb") as stream:
            raw = stream.readline(8193)
    if not raw or len(raw) > 8192:
        raise RuntimeError("Duck Mind service returned no bounded response")
    response = json.loads(raw)
    if not isinstance(response, dict) or not isinstance(response.get("ok"), bool):
        raise RuntimeError("Duck Mind service response is invalid")
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", type=Path, default=Path("/run/duckmind/duckmind.sock"))
    parser.add_argument("--request-id", default=None, help="UUID for a particular request")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    sub.add_parser("enable", help="Allow supervised play; does not enable robot motors")
    sub.add_parser("stop", help="Disable play and ask robotd to stop locomotion")
    invite = sub.add_parser("invite")
    invite.add_argument("--actor", required=True)
    invite.add_argument("--activity", choices=("look_game",), default="look_game")
    feedback = sub.add_parser("feedback")
    feedback.add_argument("interaction_id")
    feedback.add_argument("score", choices=(0, 1), type=int)
    args = parser.parse_args()
    payload: dict[str, object] = {
        "action": args.action,
        "request_id": args.request_id or str(uuid.uuid4()),
    }
    if args.action == "invite":
        payload.update(actor=args.actor, activity=args.activity)
    elif args.action == "feedback":
        payload.update(interaction_id=args.interaction_id, score=args.score)
    try:
        response = request(args.socket, payload)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(
            json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    if response["ok"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
