"""Duck Core's commands are body.sock commands (phase-4 design §11, A7; test T12).

Every command event a v4_1 duck writes (``duck.walk``, ``duck.act``, ``duck.curio``: an event
with an ``act_id``, or a ``duck.stop``) turned into its wire form ``{"cmd": …, **payload}``
must parse with phase 3's ``duckmind.duckbody.protocol.parse_command``, carry the fields
body.sock reads with the values Duck Core meant, and be exactly what its port was handed.
Turning on the spot (``vx_mm == 0``) parses; duck-body's guard refuses it from phase 5b on, so
here it is only counted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from phase4_harness import household_v2, live_v2, lively, lively_config, open_v41, walking_age

from duckmind.duckbody import protocol


def wire(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The body.sock ``cmd`` line of a command event (phase4-boundary.md §3)."""
    return {"type": "cmd", "cmd": str(payload.get("cmd", kind)), **_plain(payload)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _commands(session: Any) -> list[Any]:
    from duckmind.duckcore.acts import COMMAND_KINDS

    return [
        event
        for event in session.core.log.events
        if event.kind in COMMAND_KINDS
        and ("act_id" in event.payload or event.payload.get("cmd") == protocol.StopCmd.cmd)
    ]


def _lives() -> list[tuple[str, Any, list[dict[str, Any]]]]:
    out = []
    for seed, nest, stress in ((31, True, True), (32, False, True), (33, True, False)):
        session = open_v41(lively_config(seed, nest, stress))
        received = _record_port(session)
        walking_age(session)
        lively(session, seed, days=1)
        out.append((f"lively-{seed}", session, received))
    config, kind = household_v2(0)
    session = open_v41(config)
    received = _record_port(session)
    live_v2(session, kind, 3)
    out.append((f"v2-0-{kind}", session, received))
    return out


def _record_port(session: Any) -> list[dict[str, Any]]:
    """Every wire dict the world is handed from now on."""
    world = session.core.acts.port
    original = world.command
    received: list[dict[str, Any]] = []

    def command(commands: Any, now_ms: int) -> Any:
        received.extend(_plain(dict(c)) for c in commands)
        return original(commands, now_ms)

    world.command = command
    return received


def test_every_command_a_v41_duck_writes_parses_as_body_sock() -> None:
    turns = walks = 0
    for name, session, received in _lives():
        commands = _commands(session)
        assert commands, f"{name}: no command"
        problems = []
        for event in commands:
            line = wire(event.kind, event.payload)
            try:
                parsed = protocol.parse_command(line)
            except protocol.ProtocolError as error:
                problems.append(f"{event.event_id} {event.kind} {line}: {error}")
                continue
            payload = event.payload
            assert parsed.cmd == line["cmd"]
            if isinstance(parsed, protocol.WalkCmd):
                walks += 1
                turns += parsed.vx_mm == 0
                assert parsed.act_id == payload["act_id"]
                assert parsed.max_ms == int(payload["seconds"]) * 1000 == payload["max_ms"]
                assert parsed.deadline_ms == payload["deadline_ms"] > 0
                assert parsed.vx_mm == round(float(payload["vx"]) * 1000)
                assert parsed.vyaw_mrad == round(float(payload["vyaw"]) * 1000)
                for pose in (parsed.bel_from, parsed.expect):
                    assert pose is not None and set(pose) >= {"x_mm", "y_mm", "yaw_mrad"}
                    assert all(isinstance(pose[k], int) for k in ("x_mm", "y_mm", "yaw_mrad"))
                assert "from" not in payload and "to" not in payload
                assert "fall_ppm" not in payload
            elif isinstance(parsed, protocol.StopCmd):
                assert parsed.why == payload["why"]
            else:
                assert parsed.act_id == payload["act_id"]
                assert parsed.deadline_ms == payload["deadline_ms"]
        assert not problems, f"{name}: commands body.sock cannot read:\n" + "\n".join(problems)
        sent = [{k: v for k, v in wire(e.kind, e.payload).items() if k != "type"} for e in commands]
        assert received == sent, f"{name}: the port was not handed the commands' payloads"
    assert walks > 0
    print(f"\n[command-shapes] {walks} walk commands, {turns} of them on the spot (vx_mm = 0)")


def test_the_wire_form_of_hand_made_commands_parses() -> None:
    """The four command shapes of phase4-boundary.md §3, as Acts writes them."""
    walk = {
        "verb": "wander",
        "vx": 0.15,
        "vyaw": 0.0,
        "seconds": 4,
        "vx_mm": 150,
        "vyaw_mrad": 0,
        "max_ms": 4000,
        "purpose": "wander",
        "bel_from": {"x_mm": 0, "y_mm": 0, "yaw_mrad": 0},
        "expect": {"x_mm": 440, "y_mm": 0, "yaw_mrad": 0},
        "act_id": "A1",
        "deadline_ms": 1_790_000_009_000,
    }
    parsed = protocol.parse_command(wire("duck.walk", walk))
    assert isinstance(parsed, protocol.WalkCmd) and parsed.expect == walk["expect"]
    for payload, kind in (
        ({"cmd": "duck.posture", "want": "stand"}, protocol.PostureCmd),
        ({"cmd": "duck.dock", "bel_err_mm": 90}, protocol.DockCmd),
        (
            {
                "cmd": "duck.look",
                "kind": "seek_nest",
                "want": "nest",
                "bel_pose": {"x_mm": 300, "y_mm": -20, "yaw_mrad": 3141},
            },
            protocol.LookCmd,
        ),
    ):
        line = wire("duck.act", {**payload, "act_id": "A2", "deadline_ms": 5})
        assert isinstance(protocol.parse_command(line), kind)
    stop = protocol.parse_command(wire("duck.act", {"cmd": "duck.stop", "why": "picked_up"}))
    assert isinstance(stop, protocol.StopCmd) and stop.act_id is None
    with pytest.raises(protocol.ProtocolError):  # v4's from/to lists are not body.sock poses
        protocol.parse_command(wire("duck.walk", {**walk, "bel_from": [0, 0, 0]}))
