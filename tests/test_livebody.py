from pathlib import Path

import pytest

from duckmind.duckcore import content
from duckmind.livebody import HEAD, ClipPlayer, LivePaths


def _expected(clip: str, ms: float, amplitude: float, side: int) -> dict[str, float]:
    frame = content.frame_at(content.CLIPS[clip], ms)
    scaled = {key: value * amplitude if key != "m" else value for key, value in frame.items()}
    goal = content.head_targets(scaled, side)
    return {name: goal[name] - content.HEAD_HOME[name] for name in HEAD}


def test_the_player_sends_what_duck_core_meant():
    player = ClipPlayer(side=-1)
    player.push({"clip": "look_around", "amplitude": 6_000, "tempo": 10_000}, now=100.0)
    offsets, mouth = player.targets(100.9)
    assert offsets == pytest.approx(_expected("look_around", 900.0, 0.6, -1))
    assert mouth == 0.0


def test_tempo_stretches_time_and_the_player_goes_idle_after_the_clip():
    player = ClipPlayer()
    player.push({"clip": "nod", "amplitude": 10_000, "tempo": 20_000}, now=0.0)
    offsets, _ = player.targets(0.2)  # 400 ms of clip time at tempo 2
    assert offsets == pytest.approx(_expected("nod", 400.0, 1.0, 1))
    idle, _ = player.targets(5.0)
    assert idle == pytest.approx({name: 0.0 for name in HEAD})
    assert player.current is None


def test_held_poses_give_way_and_the_queue_stays_short():
    player = ClipPlayer()
    player.push({"clip": "nap", "amplitude": 10_000, "tempo": 10_000}, now=0.0)
    player.targets(30.0)
    assert player.current["clip"] == "nap"  # nap holds its last frame
    player.push({"clip": "perk_tilt", "amplitude": 10_000, "tempo": 10_000}, now=30.0)
    assert player.current["clip"] == "perk_tilt"
    for clip in ("nod", "preen", "look_around"):
        player.push({"clip": clip, "amplitude": 10_000, "tempo": 10_000}, now=30.1)
    assert [entry["clip"] for entry in player.queue] == ["preen", "look_around"]
    player.push({"clip": "no_such_clip"}, now=30.2)
    assert len(player.queue) == 2


def test_an_idle_glance_hands_over_to_the_next_and_never_delays_a_real_move():
    player = ClipPlayer()
    glance = {"amplitude": 6_000, "tempo": 10_000, "weak": True}
    player.push({"clip": "glance", **glance}, now=0.0)
    player.push({"clip": "look_up", **glance}, now=2.0)  # the next glance takes over
    assert player.current["clip"] == "look_up" and not player.queue
    player.push({"clip": "perk_tilt", "amplitude": 10_000, "tempo": 10_000}, now=3.0)
    assert player.current["clip"] == "perk_tilt" and not player.queue  # at once
    player.push({"clip": "glance", **glance}, now=3.5)  # never over a real move
    assert player.current["clip"] == "perk_tilt"


def test_missing_official_pieces_are_named(tmp_path):
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort.dylib")
    assert set(paths.missing()) == {
        "microduck_rl body server",
        "policies",
        "robotd",
        "ONNX Runtime",
    }
    assert LivePaths.from_env() is None or isinstance(LivePaths.from_env(), LivePaths)


def test_the_mouth_is_an_allowed_continuous_intent():
    from duckmind.body import BodyIntent

    assert BodyIntent("robot.mouth", {"open": 0.5}).params["open"] == 0.5
    with pytest.raises(ValueError):
        BodyIntent("robot.mouth", {"open": 0.5, "speed": 1})
    assert Path(__file__).name == "test_livebody.py"


def test_asleep_a_stir_plays_and_settles_back_into_the_nap_and_glances_wait():
    player = ClipPlayer()
    player.push({"clip": "nap", "amplitude": 6_000, "tempo": 10_000}, now=0.0)
    asleep, _ = player.targets(10.0)
    player.push({"clip": "glance", "amplitude": 6_000, "tempo": 10_000, "weak": True}, now=10.0)
    assert player.current["clip"] == "nap"  # an idle glance never wakes it
    player.push({"clip": "sleep_shift", "amplitude": 6_000, "tempo": 10_000}, now=10.0)
    assert player.current["clip"] == "sleep_shift"
    after, _ = player.targets(20.0)
    assert player.current["clip"] == "nap" and after == pytest.approx(asleep)
    player.wake()
    awake, _ = player.targets(21.0)
    assert awake == pytest.approx({name: 0.0 for name in HEAD})
    player.push({"clip": "glance", "amplitude": 6_000, "tempo": 10_000, "weak": True}, now=22.0)
    assert player.current["clip"] == "glance"  # awake and idle: the glance plays


def test_the_bridge_plays_each_move_at_its_moment_and_ends_on_duck_cores_posture(tmp_path):
    import time

    from duckmind.duckcore.api import Config, Session, dispatch
    from duckmind.simweb.live import LiveBridge

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths)  # never started: no processes, the queue is inspected
    bridge.ready = True
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    before = len(session.core.log.events)
    dispatch(session, "act", {"action": "owner_hello"})
    wall = time.time()
    dues = {index: wall + 30 for index in range(before, len(session.core.log.events))}
    bridge.after_call(session, before, dues=dues)
    waiting = sorted(bridge._due)
    assert waiting and all(item[0] >= time.monotonic() + 25 for item in waiting)
    posture = "stand" if session.core.s.posture in {"stand", "walk"} else "sit"
    assert waiting[-1][2] == {"sync": posture}  # Duck Core's posture, after its moves
    before = len(session.core.log.events)
    dispatch(session, "act", {"action": "pickup"})
    bridge.after_call(session, before)
    assert bridge._due and all(item[0] <= time.monotonic() + 1 for item in bridge._due)


def test_held_up_by_the_worlds_hand_the_body_sits_and_walks_wait(tmp_path):
    """Owner decision D1: while the world's hand holds the duck up (x_world hand.held) the body
    is asked to sit with its legs tucked and walks are dropped; set down, Duck Core's posture
    comes back (asked for while it was held, too)."""
    from duckmind.livebody import LiveBody

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    body = LiveBody(paths)  # never started: no processes
    body.consume([{"posture": "stand"}])
    assert body._posture_request == "stand" and body._wanted == "stand"
    body._set_held(True)
    assert body.held and body._posture_request == "sit"
    body.consume([{"walk": {"vx": 0.3, "vyaw": 0.0, "seconds": 2.0}}, {"sync": "stand"}])
    assert body._walk is None and body._posture_request == "sit"  # nothing while held
    body.drive(0.3, 0.0, 0.0)
    assert body._drive is None  # (a demonstration cannot steer a held duck either)
    body._set_held(False)
    assert not body.held and body._posture_request == "stand"  # Duck Core's posture again
    body.drive(0.3, 0.0, 0.5)
    body.skill("kick_left")
    assert body._drive is not None and body._drive[:3] == (0.3, 0.0, 0.5)
    assert body._skill == "kick_left"
    assert body.status()["held"] is False


def test_the_world_is_asked_on_its_own_connection_and_says_when_there_is_none(tmp_path):
    import time

    from duckmind.livebody import LiveBody

    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    body = LiveBody(paths, home="bodyhost")  # phase 0's body host: no world ops
    with pytest.raises(ConnectionError):
        body.world_call("x_world")
    assert body.world() is None  # (no answer yet, or a stale one: nothing to show)
    homed = LiveBody(paths)
    homed.world_state, homed.world_at = {"sim_time": 1.0}, time.monotonic()
    assert homed.world() == {"sim_time": 1.0}
    with pytest.raises(ConnectionError):
        homed.world_call("x_world")  # (not started: no body port)


def test_a_world_op_that_may_have_run_is_never_sent_twice(tmp_path, monkeypatch):
    """World ops are not idempotent (a hand script, a placement, a rolled ball): an op whose
    answer comes too late is not sent again; one on a stale connection (closed before it could
    run) gets one fresh try."""
    import json
    import socket
    import threading

    from duckmind import livebody
    from duckmind.livebody import LiveBody

    received: list[str] = []
    mode = {"slow": True}
    handled = threading.Event()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)

    def serve() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                stream = conn.makefile("rwb")
                line = stream.readline()
                if not line:
                    continue
                received.append(json.loads(line)["op"])
                if mode["slow"]:
                    threading.Event().wait(0.6)  # past the caller's timeout
                    handled.set()
                    continue  # (and then the connection closes)
                stream.write(b'{"ok": true}\n')
                stream.flush()

    threading.Thread(target=serve, daemon=True).start()
    monkeypatch.setattr(livebody, "WORLD_TIMEOUT_S", 0.3)
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    body = LiveBody(paths)
    body.body_port = listener.getsockname()[1]
    with pytest.raises(ConnectionError, match="沒有再送一次"):
        body.world_call("x_hand", do="pickup")
    assert handled.wait(5.0)
    threading.Event().wait(0.3)  # (a second copy, had one been sent, would be read by now)
    assert received == ["x_hand"]  # sent once, never again
    mode["slow"] = False
    assert body.world_call("x_world") == {"ok": True}
    listener.close()
