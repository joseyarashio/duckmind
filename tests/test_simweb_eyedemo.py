"""「用眼睛做」 (owner decision A6): the page's bridge to duck-body's body.sock."""

from __future__ import annotations

import json
import random
import re
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from duckmind.duckbody import protocol as P
from duckmind.simweb import eyedemo as E
from duckmind.tmpdir import short_tmp


def test_colour_labels_are_the_eye_s_names() -> None:
    pytest.importorskip("numpy")  # (duckmind.eye.things; the function itself is pure)
    assert E.colour_label("#f07a1f") == "orange"
    assert E.colour_label("#2f6fe0") == "blue"
    assert E.colour_label("#2fae4f") == "green"
    assert E.colour_label("#f4f1ea") == "white"
    assert E.colour_label(None) is None
    assert E.colour_label("red") is None


def test_the_thing_goes_roughly_in_front_of_the_duck() -> None:
    rng = random.Random(3)
    for yaw in (0, 1571, 3142, -1571):
        x, y = E.place_ahead({"x_mm": 100, "y_mm": -200, "yaw_mrad": yaw}, rng)
        dx, dy = x - 100, y + 200
        import math

        ahead = dx * math.cos(yaw / 1000) + dy * math.sin(yaw / 1000)
        side = -dx * math.sin(yaw / 1000) + dy * math.cos(yaw / 1000)
        assert E.PLACE_MM[0] - 2 <= ahead <= E.PLACE_MM[1] + 2
        assert abs(side) <= E.PLACE_SIDE_MM + 2


def _fake_body(path: str, script: list[dict], seen: list[dict]) -> threading.Thread:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)

    def run() -> None:
        conn, _ = server.accept()
        conn.sendall(json.dumps({"type": "hello", "api": "body-v1"}).encode() + b"\n")
        buffer = b""
        while b"\n" not in buffer:
            buffer += conn.recv(4096)
        seen.append(json.loads(buffer.split(b"\n")[0]))
        for line in script:
            conn.sendall(json.dumps(line).encode() + b"\n")
        conn.close()
        server.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_an_act_reads_its_percepts_until_its_done() -> None:
    with tempfile.TemporaryDirectory(dir=short_tmp()) as folder:
        path = str(Path(folder) / "body.sock")
        step = {"kind": "body.curio_step", "act_id": "C1", "verb": "kick", "outcome": "rolled"}
        other = {"kind": "sense.light", "level": "bright"}
        seen: list[dict] = []
        thread = _fake_body(
            path,
            [
                {"type": "ack", "act_id": "C1", "ok": True, "t_ms": 1},
                {"type": "percept", "event": other},
                {"type": "percept", "event": step},
                {"type": "done", "act_id": "C1", "kind": "body.curio_step", "t_ms": 9},
            ],
            seen,
        )
        demo = E.EyeDemo("kick", "ball_felt", E.Sockets("r", None, None, Path(folder)))
        demo.link = E.BodyLink(path)
        assert demo.link.recv(2.0)["type"] == "hello"
        command = P.CurioCmd(act_id="C1", deadline_ms=10**9, phase="step", thing="ball_felt")
        ok, why, mine = demo.act(command, 5.0)
        thread.join(2.0)
        demo.link.close()
        assert (ok, why, mine) == (True, None, [step])
        assert demo.events == [other, step]
        assert seen[0]["cmd"] == "duck.curio" and seen[0]["act_id"] == "C1"


def test_a_refused_act_says_why() -> None:
    with tempfile.TemporaryDirectory(dir=short_tmp()) as folder:
        path = str(Path(folder) / "body.sock")
        seen: list[dict] = []
        thread = _fake_body(
            path, [{"type": "ack", "act_id": "P1", "ok": False, "reason": "lifted"}], seen
        )
        demo = E.EyeDemo("peck", "ball_felt", E.Sockets("r", None, None, Path(folder)))
        demo.link = E.BodyLink(path)
        demo.link.recv(2.0)
        ok, why, _ = demo.act(P.PostureCmd(act_id="P1", deadline_ms=10**9, want="stand"), 5.0)
        thread.join(2.0)
        demo.link.close()
        assert (ok, why) == (False, "lifted")
        assert seen[0]["cmd"] == "duck.posture" and seen[0]["want"] == "stand"


def test_a_duck_that_walked_the_home_before_keeps_its_learned_floor(tmp_path: Path) -> None:
    path = E.floor_file("default_home,no_nest", tmp_path / "eye")
    assert path.name == "default_home.floor.json" and path.is_file()
    assert E.floor_ready(path)  # the seed was learned from 24 walked-clear corridors
    path.write_text(json.dumps({"walks": 1, "samples": 99}))
    assert not E.floor_ready(path)
    assert E.floor_file("default_home", tmp_path / "eye").read_text() == path.read_text()
    fresh = E.floor_file("apartment", tmp_path / "eye")  # no seed: the eye starts unlearned
    assert not fresh.exists() and not E.floor_ready(fresh)
    assert not E.floor_ready(None)


class _Spawned(Exception):
    pass


@pytest.mark.parametrize(
    "style,side,expect",
    [
        ("backin", "left", ["--dock-style", "backin", "--dock-side", "left"]),  # the apartment
        ("backin", "right", ["--dock-style", "backin", "--dock-side", "right"]),
        ("forward", None, ["--dock-style", "forward"]),
        (None, None, []),  # a home without a dock: duck-body's own default
    ],
)
def test_duck_body_is_told_the_home_s_dock_style_and_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    style: str | None,
    side: str | None,
    expect: list[str],
) -> None:
    """2026-09-28: the page's homes dock backin; the apartment's mirrored J-hook (nest.side
    left) reaches duck-body as ``--dock-side left``."""
    spawned: dict[str, list[str]] = {}

    def spawn(self: E.EyeDemo, name: str, args: list[str]) -> None:
        spawned[name] = list(args)
        raise _Spawned

    monkeypatch.setattr(E.EyeDemo, "_spawn", spawn)
    sockets = E.Sockets("r.sock", None, None, tmp_path, dock_style=style, dock_side=side)
    with pytest.raises(_Spawned):
        E.EyeDemo("kick", "ball_felt", sockets).start()
    args = spawned["duck-body"]
    dock = [i for i, a in enumerate(args) if a.startswith("--dock")]
    assert [x for i in dock for x in args[i : i + 2]] == expect


def test_the_card_09_button_asks_the_bridge_for_an_eye_demo() -> None:
    from duckmind.simweb import server

    class Bridge:
        asked: list[tuple[str, str]] = []

        def eye_demo(self, kind: str, thing: str) -> dict:
            self.asked.append((kind, thing))
            return {"kind": kind, "thing": thing, "zh": "用眼睛做：踢一下"}

    bridge = Bridge()
    answer = server.world_button(bridge, "body_eye", {"kind": "kick", "thing": "ball_felt"})  # type: ignore[arg-type]
    assert bridge.asked == [("kick", "ball_felt")]
    assert answer["note"].startswith("用眼睛做：踢一下") and "自己的眼睛" in answer["note"]
    assert "body_eye" in server.WORLD_BUTTONS
    with pytest.raises(ValueError):
        server.world_button(bridge, "body_eye", {"kind": "dance", "thing": "ball_felt"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        server.world_button(bridge, "body_eye", {"kind": "peck"})  # type: ignore[arg-type]


def test_card_09_has_the_eye_buttons_beside_the_demonstrations() -> None:
    page = (Path(E.__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    # (the words stay in the markup; the data-i18n keys beside them are for the other languages)
    page = re.sub(r'\s+data-i18n(?:-html|-attr)?="[^"]*"', "", page)
    card = page[
        page.index('id="worldTools"') : page.index("</section>", page.index('id="worldTools"'))
    ]
    for kind, zh in (("bump", "撞撞看"), ("peck", "啄一下"), ("kick", "踢一下")):
        assert f'data-world="body_eye" data-kind="{kind}">用眼睛做：{zh}</button>' in card
        assert f'data-world="body_show" data-kind="{kind}"' in card  # the 示範 stay
    assert 'class="btn accent" data-world="body_eye"' in card  # design standard: a yellow .btn
