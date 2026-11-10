"""Pollen's own LAN doors on a real duck: they go round the bridge and its deadman.

``mediad`` listens on every interface (``--host 0.0.0.0``): TCP 8443 is its WebRTC signalling and
8080 its console page, neither asks for a password and a LAN peer may drive the duck through them
(mediad ``src/route.rs``). ``netpadd`` takes a gamepad on UDP 4210 when ``[netpad] enabled`` is on
in ``robotd.toml``. Whoever comes in that way is not the brain: the bridge's token, its allowlist
and its deadman never see them.

Two looks, both read-only (a TCP connection opened and closed; for UDP a bind tried on the duck
itself — nothing is ever sent to mediad or netpadd):

* **on the duck**, when a real duck's bridge starts (:func:`look`): it knocks on the duck's own
  LAN address. An open door answers; a door closed by ``tools/duck/close-mediad.sh`` (which
  accepts loopback *addresses* only, so a connection from the duck to its own LAN address is
  dropped like a LAN peer's), by another firewall, or by mediad bound to ``127.0.0.1`` does not.
  A UDP door cannot be knocked on, so netpadd is "open" whenever something holds its port: from
  the duck the bridge cannot see whether a firewall drops it. The bridge refuses to serve while a
  door is open (:func:`refusal`), unless it is started with ``--allow-open-mediad``;
* **from the computer** (``remotebody check``, :func:`open_ports`): the same TCP knock from where
  a LAN peer stands, which is the look that counts; ``check`` does not call the link ready while
  a door answers.
"""

from __future__ import annotations

import errno
import ipaddress
import socket
from dataclasses import dataclass
from typing import Any

MEDIAD_PORTS = (8443, 8080)  # mediad/src/main.rs: --port (signalling), --web-port (console)
NETPADD_PORT = 4210  # netpadd: a gamepad over UDP ([netpad] enabled, off as shipped)
PROBE_S = 1.0
CLOSE_SCRIPT = "tools/duck/close-mediad.sh"
BOARD_DOC = "docs/deploy/board.md, 'Closing mediad's LAN doors'"


def open_ports(
    host: str, ports: list[int] | tuple[int, ...], timeout_s: float = PROBE_S
) -> list[int]:
    """The ``ports`` on ``host`` that accept a TCP connection from this machine."""
    found = []
    for port in ports:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                found.append(port)
        except OSError:
            pass
    return found


def udp_held(host: str, port: int) -> bool:
    """Something on this machine holds UDP ``port`` at ``host`` (its own address): a bind there
    is refused as in use. Nothing is sent."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as error:
            return error.errno == errno.EADDRINUSE
    return False


def own_lan_address() -> str | None:
    """This machine's address on its default route (what a LAN peer dials); None without one.
    A UDP socket is pointed at a documentation address (RFC 5737): nothing is sent."""
    for family, far in ((socket.AF_INET, "192.0.2.1"), (socket.AF_INET6, "2001:db8::1")):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect((far, 9))
                address = ipaddress.ip_address(str(sock.getsockname()[0]))
        except (OSError, ValueError):
            continue
        if not (address.is_loopback or address.is_unspecified):
            return str(address)
    return None


@dataclass(frozen=True)
class Doors:
    """What a look found. ``host`` None: the duck's LAN address is not known, nothing was
    knocked on (so nothing is known to be closed)."""

    host: str | None
    tcp: tuple[int, ...] = ()  # mediad's ports that answered
    netpadd: int | None = None  # netpadd's UDP port when something holds it

    @property
    def open(self) -> bool:
        return self.host is None or bool(self.tcp) or self.netpadd is not None

    def to_json(self) -> dict[str, Any]:
        return {"host": self.host, "tcp_open": list(self.tcp), "netpadd": self.netpadd}


def look(
    host: str | None,
    tcp_ports: list[int] | tuple[int, ...] = MEDIAD_PORTS,
    udp_port: int | None = NETPADD_PORT,
    timeout_s: float = PROBE_S,
) -> Doors:
    """On the duck: knock on its own LAN address ``host`` (see the module's words)."""
    if host is None:
        return Doors(None)
    held = udp_port is not None and udp_held(host, udp_port)
    return Doors(host, tuple(open_ports(host, tcp_ports, timeout_s)), udp_port if held else None)


def refusal(doors: Doors, *, allowed: bool) -> list[str]:
    """The bridge's words when a door is open: what, why it matters, and the fix."""
    out = []
    if doors.host is None:
        out.append(
            "cannot find this duck's LAN address to knock on mediad's doors: give it with "
            "--lan-address (ip -4 addr)"
        )
    if doors.tcp:
        ports = ", ".join(str(p) for p in doors.tcp)
        out.append(
            f"mediad's LAN doors are OPEN: {doors.host} TCP port {ports} (no password: anyone on "
            "this network can drive the duck through them, around this bridge and its deadman)"
        )
        out.append(
            f"  close them on the duck: sudo sh {CLOSE_SCRIPT} --brain <the computer's address> "
            f"({BOARD_DOC})"
        )
    if doors.netpadd is not None:
        out.append(
            f"netpadd's LAN door is OPEN: something listens on UDP {doors.netpadd} (a gamepad on "
            "the network drives the duck around this bridge)"
        )
        out.append(
            "  turn it off: [netpad] enabled = false (sudo robotctl configure, then restart "
            f"robotd). {CLOSE_SCRIPT} drops it for the LAN too, but from the duck the bridge "
            "cannot see that"
        )
    if allowed:
        out.append("serving anyway (--allow-open-mediad): the deadman does not cover those doors")
    else:
        out.append(
            "the bridge does not serve while they are open. Close them and start it again, or "
            "start it with --allow-open-mediad to serve anyway"
        )
    return out


__all__ = [
    "BOARD_DOC",
    "CLOSE_SCRIPT",
    "MEDIAD_PORTS",
    "NETPADD_PORT",
    "Doors",
    "look",
    "open_ports",
    "own_lan_address",
    "refusal",
    "udp_held",
]
