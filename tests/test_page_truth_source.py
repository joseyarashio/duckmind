"""Where the page's duck comes from: phase 0 「看見真相」 (spec §13, §17 期0; G15, W05, W07, W10)
and phase 2 「真的家與真的感測器」 (spec §17 期2; owner decisions A1, D1, D2, B4).

With live physics the solid duck on the page is the physics truth only — the body frame's
``truth`` (where the body is in the world: phase 2's home, whose coordinates in
``default_home`` are Duck Core's own room) with its trunk height, quaternion and joints. Duck
Core's plan (its pose and moves, moved into the world by the home's frame) may move only the
shadow duck. Without truth (a fast-forward, no live physics) the solid duck is the plan and the
page says so.

Phase 2 changed what moves the body besides its own physics, and these tests with it:

* phase 0 *seated the physics world* at Duck Core's pose (anchors "start", "sync", "hand")
  without ever moving the body. Now the world has fixed walls, so the body itself is **placed**
  (``x_place``, a world operation the page labels): where Duck Core is when the body starts,
  and where its plan is once a fast-forward stops. The settling tests of phase 0 (the world
  re-seated while the body sat down) are gone: a placed body is set down at rest, and whatever
  it does afterwards is the truth.
* phase 0's interim hand (an anchor ``hand``: the world seated where a hand put the duck) is
  replaced by **the world's hand** (``x_hand``): Duck Core's hand events become hand scripts
  that really carry the body (D1), unless a family member's hand already did it; 「扶起物理身體」
  is the hand too (D2).

The behavioural tests run ``static/truth.js`` and ``static/home3d.js`` (with the page's own
three.js, on the home the server sends) in node, and the server's bridge with a stand-in body that
answers the world ops (no robotd, no MuJoCo).
"""

from __future__ import annotations

import heapq
import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from duckmind.duckcore.api import Config, Session, _motion, dispatch, open_session
from duckmind.livebody import LivePaths
from duckmind.simweb.live import (
    INTERRUPTS,
    HomeFrame,
    LiveBridge,
    done_already,
    hand_command,
    heading_of,
    physics_pose,
    spiral,
    yaw_of,
)

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
TRUTH_JS = STATIC / "truth.js"
HOME3D_HARNESS = Path(__file__).with_name("home3d_node_harness.mjs")


def _zh_words() -> dict[str, str]:
    """The page's 繁體中文 words by catalog key (static/i18n.zh-Hant.*.js; docs/i18n.md)."""
    words: dict[str, str] = {}
    for path in sorted(STATIC.glob("i18n.zh-Hant.*.js")):
        body = path.read_text(encoding="utf-8")
        words.update(
            json.loads(body[body.index("{", body.index("DuckI18n.load(")) : body.rindex("}") + 1])
        )
    return words


# ── behaviour: static/truth.js in node ────────────────────────────────────────────


NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(script: str) -> Any:
    source = f'import * as T from "{TRUTH_JS.as_uri()}";\n{script}'
    done = subprocess.run(
        [str(NODE), "--input-type=module", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


FRAME = """
const legs = { left_knee: 1.1, right_knee: -1.2, left_hip_pitch: -0.4 };
const head = { neck_pitch: 0.31, head_pitch: 0.29, head_yaw: 0.02, head_roll: -0.01 };
const frame = (x, y, yaw, placed = 111) => ({
  t: 3, trunk: [x, y, 0.118], quat: [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)],
  joints: { ...legs, ...head },
  truth: { x_mm: Math.round(x * 1000), y_mm: Math.round(y * 1000), yaw_mrad: Math.round(yaw * 1000), frame: "world", placed_at: placed, placed_why: "start" },
});
const plan = (x, y, yaw, knee = 0.3) => ({
  joints: { left_knee: knee, right_knee: -knee, left_hip_pitch: 0, neck_pitch: 0.9, head_pitch: -0.7, head_yaw: 0.4, head_roll: 0.2 },
  trunk: [x, y, 0.06], quat: [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], mouth: 0.5,
});
const solidOf = (f, p, physicsHead = true) => {
  const target = T.truthTarget(f);
  return T.compose({ plan: p, truth: T.follow(null, target, 0.016), physicsHead });
};
"""


@needs_node
def test_in_live_mode_the_duck_is_where_the_truth_says_and_the_plan_moves_only_the_shadow():
    out = _node(
        FRAME
        + """
const f = frame(1.2, -0.4, 0.5);
const plans = [plan(0, 0, 0), plan(3, -2.5, 2.9, 0.9), plan(-1, 1, -1.2), plan(1.2, -0.4, 0.5)];
const drawn = plans.map((p) => solidOf(f, p));
const moved = [frame(0.3, 0.9, -2.0), frame(-2.5, 0.1, 3.0)].map((g) => solidOf(g, plans[0]));
const noTruth = [T.compose({ plan: plans[1], truth: T.truthTarget({ ...f, truth: null }) }),
                 T.compose({ plan: plans[1], truth: T.truthTarget(null) })];
console.log(JSON.stringify({
  modes: drawn.map((d) => d.mode),
  solids: drawn.map((d) => d.solid),
  shadows: drawn.map((d) => d.shadow),
  plans,
  moved: moved.map((d) => ({ solid: d.solid, shadow: d.shadow })),
  noTruth: noTruth.map((d) => ({ mode: d.mode, solid: d.solid, shadow: d.shadow })),
  yaw: drawn.map((d) => T.yawOf(d.solid.quat)),
  headOff: solidOf(f, plans[1], false).solid.joints,
}));
"""
    )
    assert out["modes"] == ["truth"] * 4
    first = out["solids"][0]
    assert first["trunk"][:2] == pytest.approx([1.2, -0.4])
    assert first["trunk"][2] == pytest.approx(0.118)  # the physics trunk height
    assert all(yaw == pytest.approx(0.5, abs=1e-9) for yaw in out["yaw"])
    for solid in out["solids"][1:]:
        # Whatever Duck Core plans, the solid duck does not move, turn or bend.
        assert solid["trunk"] == pytest.approx(first["trunk"])
        assert solid["quat"] == pytest.approx(first["quat"])
        assert solid["joints"] == pytest.approx(first["joints"])
    assert first["joints"]["left_knee"] == pytest.approx(1.1)  # the physics legs
    assert out["shadows"] == out["plans"]  # the plan goes to the shadow, as it is
    # The truth moves the solid duck and never the shadow.
    assert [m["solid"]["trunk"][:2] for m in out["moved"]] == [
        pytest.approx([0.3, 0.9]),
        pytest.approx([-2.5, 0.1]),
    ]
    assert all(m["shadow"] == out["plans"][0] for m in out["moved"])
    # Without truth there is one duck: the plan itself, and no shadow.
    for item in out["noTruth"]:
        assert item["mode"] == "plan" and item["solid"] == out["plans"][1]
        assert item["shadow"] is None
    # "頭部也用物理" off: only the four head joints show Duck Core's command.
    head_off = out["headOff"]
    assert head_off["neck_pitch"] == pytest.approx(0.9) and head_off["head_yaw"] == pytest.approx(
        0.4
    )
    assert head_off["left_knee"] == pytest.approx(1.1)


@needs_node
def test_the_drawn_truth_eases_but_a_world_placement_is_shown_at_once():
    out = _node(
        FRAME
        + """
let shown = T.follow(null, T.truthTarget(frame(0, 0, 0)), 0.016);
shown = T.follow(shown, T.truthTarget(frame(0.02, 0, 0)), 0.016);
const eased = shown.x;
const placed = T.follow(shown, T.truthTarget(frame(0.03, 0, 0, 222)), 0.016);
const jumped = T.follow(placed, T.truthTarget(frame(1.2, 0, 0, 222)), 0.016);
// A hand carrying the body (no placement, a few centimetres a frame) is followed, not snapped.
let carried = T.follow(null, T.truthTarget(frame(0, 0, 0, 333)), 0.016);
const path = [];
for (let i = 1; i <= 10; i++) {
  carried = T.follow(carried, T.truthTarget(frame(0.04 * i, 0, 0, 333)), 0.016);
  path.push(carried.x);
}
console.log(JSON.stringify({ eased, placed: placed.x, jumped: jumped.x, path }));
"""
    )
    assert 0 < out["eased"] < 0.02  # a small move eases in
    assert out["placed"] == pytest.approx(0.03)  # a world placement is shown at once
    assert out["jumped"] == pytest.approx(1.2)  # so is a jump (never a slide across the room)
    path = out["path"]
    assert all(b > a for a, b in zip(path, path[1:], strict=False))  # followed all the way…
    assert path[-1] < 0.4  # …easing behind the carried body, never teleported


@needs_node
def test_the_truth_trail_keeps_ten_minutes_and_breaks_where_the_body_was_placed():
    out = _node(
        """
const trail = [];
for (let i = 0; i < 60; i++) T.trailPush(trail, { x: i * 0.02, y: 0, a: 1 }, i * 100);
for (let i = 0; i < 20; i++) T.trailPush(trail, { x: 5 + i * 0.02, y: 0, a: 2 }, 6000 + i * 100);
const now = 8000;
const all = T.trailSegments(trail, now, 0);
const crossing = all.filter(([x0, , x1]) => x0 < 2 && x1 > 4);
T.trailPush(trail, { x: 9, y: 9, a: 2 }, 11 * 60 * 1000);
const still = T.trailPush(trail, { x: 9.001, y: 9, a: 2 }, 11 * 60 * 1000 + 100);
const sway = [];
for (let i = 0; i < 40; i++) T.trailPush(sway, { x: i * 0.02, y: 0.02 * Math.sin(i * 2.1), a: 1 }, i * 60);
const raw = Math.max(...T.trailSegments(sway, 2400, 0).map((s) => Math.abs(s[1])));
const smooth = Math.max(...T.trailSegments(sway, 2400).map((s) => Math.abs(s[1])));
console.log(JSON.stringify({ segments: all.length, crossing: crossing.length, left: trail.length, still, raw, smooth }));
"""
    )
    assert out["segments"] == 59 + 19 and out["crossing"] == 0
    assert out["left"] == 1 and out["still"] is False  # older than 10 minutes: gone
    assert out["smooth"] < out["raw"] / 3  # the drawn path smooths the gait's sway


@needs_node
def test_the_label_says_which_duck_is_drawn():
    out = _node(
        """
const base = { enabled: true, on: true, available: true };
console.log(JSON.stringify({
  truth: T.modeLabel({ ...base, truth: true }),
  home: T.modeLabel({ ...base, truth: true, physicalHome: true }),
  held: T.modeLabel({ ...base, truth: true, physicalHome: true, held: true }),
  fall: T.modeLabel({ ...base, truth: true, planFallen: true }),
  down: T.modeLabel({ ...base, truth: true, physicsFallen: true }),
  fast: T.modeLabel({ ...base, truth: false, fastForward: true }),
  none: T.modeLabel({ truth: false, enabled: false }),
  off: T.modeLabel({ ...base, truth: false, on: false }),
  broken: T.modeLabel({ ...base, truth: false, available: false, broken: true }),
  starting: T.modeLabel({ ...base, truth: false, available: false }),
  old: T.modeLabel({ ...base, truth: false, oldServer: true }),
  seating: T.modeLabel({ ...base, truth: false }),
}));
"""
    )
    # (the label's words are catalog keys: read in 繁體中文, the page's source language)
    words = _zh_words()
    out = {
        name: {**label, "title": words[label["title"]], "detail": words[label["detail"]]}
        for name, label in out.items()
    }
    # Phase 0's flat floor: the drawn room is only on the page, and it says so.
    assert out["truth"]["title"] == "物理真相" and "只有畫面，物理裡沒有" in out["truth"]["detail"]
    # Phase 2's home: nothing is marked "only on the page" any more.
    assert "只有畫面" not in out["home"]["detail"] and "物理裡都是真的" in out["home"]["detail"]
    assert "手" in out["held"]["detail"] and "扶起物理身體" in out["down"]["detail"]
    assert "骰子" in out["fall"]["detail"]
    for key in ("fast", "none", "off", "broken", "starting", "old", "seating"):
        assert out[key]["mode"] == "plan"
        assert out[key]["title"] == "這是小鴨的計畫，不是物理"
    assert "快轉" in out["fast"]["detail"] and "--live" in out["none"]["detail"]
    # A broken body is never called "starting" (robotd died, the body server stopped).
    assert "出錯" in out["broken"]["detail"] and "啟動中" not in out["broken"]["detail"]
    assert "啟動中" in out["starting"]["detail"]
    assert "舊版" in out["old"]["detail"] and "舊版" not in out["seating"]["detail"]


# ── behaviour: static/home3d.js in node, on the home the server sends ───────────────

CAMERA_OUTSIDE = [0.0, -6.0, 1.5]  # south of the home, looking in at the duck (world metres)
CAMERA_INSIDE = [0.0, -2.0, 1.5]
DUCK_AT = [0.0, 0.0, 0.1]
TRUNK = {"trunk": [1.2, -0.4, 0.118], "quat": [1.0, 0.0, 0.0, 0.0]}  # the drawn solid duck
WORLD = {  # (/api/body's `world`: a toy that rolled away, a hand holding the duck, a person)
    "things": [{"id": "ball_felt", "pos_mm": [1500, 900, 35], "quat": [1, 0, 0, 0]}],
    "hand": {"shown": True, "holding": True, "pos_mm": [500, 500, 300], "quat": [1, 0, 0, 0]},
    "bodies": {"person_owner": {"pos_mm": [2000, 1000, 0], "quat": [1, 0, 0, 0]}},
}


@pytest.fixture(scope="module")
def home() -> dict[str, Any]:
    """What /api/home answers for the default home (the layout the physics runs)."""
    from duckmind.simweb.homeview import home_payload

    return home_payload("default_home")


@pytest.fixture(scope="module")
def built(home: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if NODE is None:
        pytest.skip("node is not installed")
    looks = [{"camera": camera, "duck": DUCK_AT} for camera in (CAMERA_OUTSIDE, CAMERA_INSIDE)]
    path = tmp_path_factory.mktemp("home3d") / "input.json"
    path.write_text(
        json.dumps({"payload": home, "looks": looks, "world": WORLD, "trunk": TRUNK}),
        encoding="utf-8",
    )
    done = subprocess.run(
        [str(NODE), str(HOME3D_HARNESS), str(STATIC), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_the_home_is_drawn_from_the_layout_the_physics_runs(home, built):
    """The physical room: home3d.js draws every element of the layout the page may show, where
    the physics has it, with the toys, the hand, the family's bodies and the dock's real
    AprilTag cells; a home the physics does not run is drawn translucent (W07)."""
    layout = home["layout"]
    on_page = [el for el in layout["elements"] if el["flags"]["page"]]
    assert len(on_page) < len(layout["elements"])  # (camera-only bands and the ceiling: not drawn)
    assert [el["id"] for el in built["elements"]] == [el["id"] for el in on_page]
    drawn = {el["id"]: el for el in built["elements"]}
    for el in on_page:
        x0, y0, z0, x1, y1, z1 = drawn[el["id"]]["box"]
        # over its place on the floor, standing on its lowest point, as big as it is
        assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx(el["pos_mm"][:2], abs=1), el["id"]
        assert z0 == pytest.approx(el["pos_mm"][2], abs=1), el["id"]
        if el["shape"] == "box" and not el["yaw_mrad"]:
            assert [x1 - x0, y1 - y0, z1 - z0] == pytest.approx(el["size_mm"], abs=1), el["id"]
    assert [(toy["id"], toy["at"], toy["parts"]) for toy in built["things"]] == [
        (toy["id"], toy["pos_mm"], len(toy["parts"])) for toy in layout["things"]
    ]
    # the hand and the family's bodies are there, hidden until the world shows them
    assert built["hand"] == {"parts": len(home["hand"]["parts"]), "visible": False}
    bodies = [body for person in home["people"].values() for body in person["bodies"]]
    assert [body["body"] for body in built["people"]] == bodies
    assert not any(body["visible"] for body in built["people"])
    # the dock's tags have the same bits as the physics' texture
    tags = home["nest_parts"]["tags"]
    assert len(tags) == 2 and built["tags"] == [tag["cells"] for tag in tags]
    assert built["contacts"] == [contact["id"] for contact in home["nest_parts"]["contacts"]]
    # only on the page (the physics runs another world): translucent, all but the floor
    for el in on_page:
        assert built["solid"][el["id"]] == el["opacity"], el["id"]
        if el["kind"] == "floor":
            assert built["ghost"][el["id"]] == 1
        elif el["kind"] != "wall":  # (the walls' opacity is the fading's, below)
            assert built["ghost"][el["id"]] < 0.3, el["id"]


@needs_node
def test_walls_in_the_way_fade_and_the_world_puts_its_things_where_they_are(home, built):
    """B4: a wall between the camera and the duck fades, and comes back when it is no longer
    between. The toys, the family and the hand are drawn where the world's truth has them; a
    hand that holds the duck is drawn on the drawn duck's own trunk (no lag behind the body)."""
    walls = [el for el in home["layout"]["elements"] if el["kind"] == "wall"]
    between = {
        wall["id"]
        for wall in walls  # (the look runs north along x = 0, from the camera to the duck)
        if abs(wall["pos_mm"][0]) <= wall["size_mm"][0] / 2
        and CAMERA_OUTSIDE[1] * 1000 < wall["pos_mm"][1] < DUCK_AT[1] * 1000
    }
    assert len(between) == 1 and len(walls) >= 4
    outside, inside = built["looks"]
    assert set(outside) == {wall["id"] for wall in walls}
    for wall, seen in outside.items():
        if wall in between:
            assert seen["faded"] and seen["transparent"] and seen["opacity"] < 0.2
        else:
            assert not seen["faded"] and not seen["transparent"] and seen["opacity"] == 1
    for seen in inside.values():  # the camera came inside: no wall is in the way
        assert not seen["faded"] and not seen["transparent"] and seen["opacity"] == 1
    held, released, gone = built["held"], built["released"], built["gone"]
    assert held["things"] == {"ball_felt": [1500, 900, 35]}
    assert held["hand"] == {"visible": True, "at": [1200, -400, 118]}  # on the duck's trunk
    here = {body: at for body, at in held["people"].items() if at is not None}
    assert here == {"person_owner": [2000, 1000, 0]}
    assert released["hand"] == {"visible": True, "at": [500, 500, 300]}  # the hand's own place
    assert not gone["hand"]["visible"] and not any(gone["people"].values())


def test_every_world_button_on_the_page_asks_for_something_the_server_does():
    """D2 「扶起物理身體」 and card 09 (示範 and 用眼睛做 bump/peck/kick, roll a ball, tidy up):
    what each button of the page sends — its name, its kind and the chosen toy — is a call the
    server's world buttons take, and it reaches the world (never Duck Core, W05)."""
    from page_tree_node_harness import elements_with

    from duckmind.simweb.server import WORLD_BUTTONS, world_button

    class World:
        neuro = None

        def __init__(self) -> None:
            self.asked: list[tuple[Any, ...]] = []

        def upright(self) -> dict[str, Any]:
            self.asked.append(("upright",))
            return {"ok": True, "eta_s": 2.5}

        def demo(self, kind: str, thing: str, side: str | None) -> dict[str, Any]:
            self.asked.append(("demo", kind, thing))
            return {"ok": True}

        def eye_demo(self, kind: str, thing: str) -> dict[str, Any]:
            self.asked.append(("eye_demo", kind, thing))
            return {"ok": True}

        def world_action(self, op: str, **params: Any) -> dict[str, Any]:
            self.asked.append((op, params))
            return {"ok": True}

    world = World()
    toy = elements_with("value", inside="worldThing")[0]["value"]  # the chooser's first toy
    for button in elements_with("data-world"):
        assert button["data-world"] in WORLD_BUTTONS, button
        sent = {"kind": button.get("data-kind", ""), "thing": toy}  # (app.js: the button's click)
        assert world_button(world, button["data-world"], sent)["note"], button  # type: ignore[arg-type]
    assert world.asked == [
        *[("demo", kind, toy) for kind in ("bump", "peck", "kick")],
        *[("eye_demo", kind, toy) for kind in ("bump", "peck", "kick")],
        ("x_roll_ball", {"toward": "duck", "id": toy}),
        ("x_tidy", {"to": "toy_corner"}),
        ("upright",),
    ]


# ── behaviour: the server's truth, placements and the hand, with a stand-in body ────


class StandInBody:
    """A physics body without processes: frames at a pose we set, and a world that answers
    ``x_place`` (it moves the body there, unless the spot is inside ``blocked``), ``x_hand``
    (it records the request) and ``x_world``."""

    error: str | None = None
    host_info: dict[str, Any] | None = {"ops": ["x_place", "x_world", "x_hand", "x_people"]}
    home: str | None = "default_home"
    policy = "sit"

    def __init__(self) -> None:
        self.pose = (0.0, 0.0, 0.0)
        self.consumed: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.blocked: list[tuple[float, float, float]] = []  # (x, y, radius) nothing may enter

    def put(self, x: float, y: float, yaw: float) -> None:
        self.pose = (x, y, yaw)

    def frame(self) -> dict[str, Any]:
        x, y, yaw = self.pose
        return {
            "t": 1.0,
            "joints": {"left_knee": 1.0},
            "trunk": [x, y, 0.06],
            "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
            "policy": self.policy,
            "fallen": False,
            "realtime": 1.0,
        }

    def status(self) -> dict[str, Any]:
        return {
            "running": True,
            "policy": "sit",
            "fallen": False,
            "realtime": 1.0,
            "error": None,
            "monitor": {"trust": "ok", "why_zh": []},
        }

    def world(self) -> dict[str, Any]:
        return {"home": {"name": "default_home", "source": "layout", "fingerprint": "f"}}

    def world_call(self, op: str, **params: Any) -> dict[str, Any]:
        self.calls.append((op, dict(params)))
        if op == "x_place":
            x, y = params["x_mm"] / 1000, params["y_mm"] / 1000
            for bx, by, radius in self.blocked:
                if math.hypot(x - bx, y - by) < radius:
                    return {"ok": False, "blocked_by": ["home_sofa_seat"], "why_zh": ["擋住了"]}
            self.put(x, y, params.get("yaw_mrad", 0) / 1000)
            return {"ok": True, "x_mm": params["x_mm"], "y_mm": params["y_mm"], "why_zh": []}
        if op == "x_hand":
            return {"ok": True, "script": len(self.calls), "eta_s": 2.5, "why_zh": []}
        return {"ok": True}

    def hands(self) -> list[dict[str, Any]]:
        return [params for op, params in self.calls if op == "x_hand"]

    def consume(self, motions: list[dict[str, Any]]) -> None:
        self.consumed.extend(motions)

    def stop(self) -> None:
        return None


def _bridge(tmp_path: Path, body: StandInBody | None = None) -> tuple[LiveBridge, StandInBody]:
    paths = LivePaths(tmp_path, tmp_path, tmp_path, tmp_path / "robotd", tmp_path / "ort")
    bridge = LiveBridge(paths, eye=False)  # never started: no processes, no camera
    bridge.sync_quiet_s = 0  # a fast-forward's placement at once (its waiting has its own test)
    body = body or StandInBody()
    bridge.body = body  # type: ignore[assignment]
    bridge.ready = True
    return bridge, body


def _truth(bridge: LiveBridge) -> dict[str, Any]:
    truth = bridge.truth(bridge.body.frame())
    assert truth is not None
    return truth


def _play(bridge: LiveBridge) -> None:
    """Run what is due now (motions, hand scripts) the way the scheduler thread does."""
    worker = threading.Thread(target=bridge._play_due, daemon=True)
    worker.start()
    time.sleep(0.3)
    bridge._closing = True
    with bridge._wake:
        bridge._wake.notify()
    worker.join(timeout=2)
    bridge._closing = False


def test_the_truth_is_the_physics_in_the_world_and_a_start_places_the_body_where_the_duck_is(
    tmp_path,
):
    bridge, body = _bridge(tmp_path)
    body.put(0.03, -0.01, 0.2)  # the body at its spawn, in the world
    before = _truth(bridge)  # the truth is the physics, wherever it is (world coordinates)
    assert (before["x_mm"], before["y_mm"], before["yaw_mrad"]) == (30, -10, 200)
    assert before["frame"] == "world" and before["placed_at"] is None
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    s = session.core.s
    s.pos_x, s.pos_y, s.heading = 1200, -400, 1571  # Duck Core thinks: 1.2 m, −0.4 m, facing +y
    bridge.resync(session, "start")  # a world operation: x_place at Duck Core's pose
    place = [params for op, params in body.calls if op == "x_place"]
    assert place and (place[0]["x_mm"], place[0]["y_mm"], place[0]["yaw_mrad"]) == (
        1200,
        -400,
        1571,
    )
    assert place[0]["keyframe"] == "SIT"
    first = _truth(bridge)
    assert (first["x_mm"], first["y_mm"], first["yaw_mrad"]) == (1200, -400, 1571)
    assert first["placed_why"] == "start" and first["placed_at"] > 0
    # The body walks 0.5 m straight ahead (+y in the world) and turns 0.3 rad: that is the truth.
    body.put(1.2, 0.1, 1.871)
    walked = _truth(bridge)
    assert (walked["x_mm"], walked["y_mm"], walked["yaw_mrad"]) == (1200, 100, 1871)
    # Duck Core changing its mind (its plan, a teleport onto the mat) moves nothing here.
    s.pos_x, s.pos_y, s.heading = 0, 0, 0
    assert _truth(bridge) == walked
    physics = physics_pose(body.frame())
    assert physics is not None and physics[0] == pytest.approx(1.2)
    # The body is then told Duck Core's power and posture.
    kinds = [item[2] for item in sorted(bridge._due)]
    assert {"power": "on"} in kinds and {"sync": "sit"} in kinds


def test_a_spot_that_is_taken_moves_to_the_nearest_free_one(tmp_path):
    bridge, body = _bridge(tmp_path)
    body.blocked = [(0.9, 2.45, 0.25)]  # (the sofa)
    placed = bridge.place((0.9, 2.45, 0.0), "sync")
    assert placed["ok"] and placed["blocked_by"] == ["home_sofa_seat"]
    assert 250 <= placed["moved_mm"] <= 350 and placed["why_zh"]  # (the words say so)
    assert math.hypot(body.pose[0] - 0.9, body.pose[1] - 2.45) >= 0.25
    body.blocked = [(0.0, 0.0, 5.0)]  # nowhere within reach
    stuck = bridge.place((0.0, 0.0, 0.0), "sync")
    assert not stuck["ok"] and "被擋住" in stuck["why_zh"][0]
    spots = list(spiral(0.0, 0.0))
    assert spots[0] == (0.0, 0.0) and max(math.hypot(*spot) for spot in spots) == pytest.approx(0.8)
    # Phase 0's body host has no world: nothing is moved, and the note says why.
    bridge.body.home = None  # type: ignore[union-attr]
    flat = bridge.place((1.0, 1.0, 0.0), "start")
    assert not flat["ok"] and "沒有物理世界" in flat["why_zh"][0]


def test_hand_events_become_the_worlds_hand_and_nothing_else_moves_the_body(tmp_path):
    """D1 (replaces phase 0's hand anchor): a duck with a nest is lifted out and held, put on
    the floor where Duck Core says, picked up and put back into its nest — each by the world's
    hand, which really carries the body (the truth follows the physics, never jumps). Switching
    it off or the end of a demonstration moves nothing (W05)."""
    bridge, body = _bridge(tmp_path)
    session = Session(Config(seed=7, nest=True))
    dispatch(session, "advance", {"until": "10:00"})
    bridge.resync(session, "start")
    body.put(0.4, 0.2, 1.0)  # the body walked off somewhere
    walked = _truth(bridge)
    events = session.core.log.events
    for step in (
        ("set_setting", {"key": "floor", "value": False}),  # lifted out of the nest…
        ("set_setting", {"key": "floor", "value": True}),  # …and set down on the floor
        ("act", {"action": "pickup", "params": {}}),
        ("act", {"action": "place_on_nest", "params": {}}),
    ):
        at = len(events)
        dispatch(session, *step)
        bridge.after_call(session, at)  # played now, as the page's buttons are
    _play(bridge)
    kinds = {event.kind for event in events}
    assert {"env.floor", "nest.docked", "sense.pickup"} <= kinds
    hands = body.hands()
    dos = [(hand["do"], hand.get("to")) for hand in hands]
    assert dos[-1] == ("place", "nest") and ("pickup", None) in dos
    floor = next(hand for hand in hands if hand.get("to") == "floor")
    # "On the floor" is just in front of the nest (Duck Core's plan), turned into the world.
    assert (floor["x_mm"], floor["y_mm"], floor["yaw_mrad"]) == (450, 0, 0)
    # The hand carries the body in the physics; here the stand-in did not move: no jump.
    assert _truth(bridge) == walked
    at = len(events)
    dispatch(session, "act", {"action": "power_off", "params": {}})
    bridge.after_call(session, at)
    bridge.demo_until = time.time() - 1  # a demonstration that just ended
    _play(bridge)
    assert len(body.hands()) == len(hands)  # switched off, demo over: no hand
    assert _truth(bridge) == walked


def test_only_a_hand_counts_as_a_hand():
    """hand_command: a pickup holds a nest duck up and puts one without a nest back on its desk
    (the mat, A2/A3); env.floor carries its pose; a hand puts it into its nest; the duck walking
    in by itself (or re-seating itself) is only what it believes (W05)."""

    class Ev:
        def __init__(self, kind: str, **payload: Any) -> None:
            self.kind, self.payload = kind, payload

    nest, flat = {"nest": True}, {"nest": False}
    assert hand_command(Ev("sense.pickup"), **nest) == {"do": "pickup"}
    assert hand_command(Ev("sense.pickup"), **flat) == {"do": "place", "to": "desk"}
    down = Ev("env.floor", floor=True, why="family", pose=[0.25, -0.1, 1.5])
    assert hand_command(down, **nest) == {"do": "place", "to": "floor", "pose": [0.25, -0.1, 1.5]}
    assert hand_command(Ev("env.floor", floor=False, why="pickup"), **nest) is None  # (held)
    assert hand_command(Ev("env.floor", floor=False, why="pickup"), **flat) == {
        "do": "place",
        "to": "desk",
    }
    assert hand_command(Ev("env.floor", floor=False, why="family"), **flat)["to"] == "desk"
    assert hand_command(Ev("nest.docked", by="hand"), **nest) == {"do": "place", "to": "nest"}
    assert hand_command(Ev("nest.docked", by="family"), **nest) == {"do": "place", "to": "nest"}
    assert hand_command(Ev("nest.docked", by="self"), **nest) is None
    assert hand_command(Ev("nest.docked", by="reseat"), **nest) is None
    assert hand_command(Ev("sense.set_upright"), **nest) == {"do": "upright"}
    assert hand_command(Ev("duck.walk", seconds=3), **nest) is None


def test_what_a_family_members_hand_did_is_not_done_twice(tmp_path):
    """D3: the family visit's hand already put the duck into its nest before Duck Core heard
    about it (``done_by_hand``): the bridge does not carry it there a second time. A duck
    without a nest that the visit picked up is still set back on its desk."""
    assert done_already({"do": "place", "to": "nest"}, {"hand": "place", "to": "nest"})
    assert not done_already({"do": "place", "to": "floor"}, {"hand": "place", "to": "nest"})
    assert done_already({"do": "upright"}, {"hand": "upright"})
    assert done_already({"do": "pickup"}, {"hand": "pickup", "holding": True})
    assert not done_already({"do": "place", "to": "desk"}, {"hand": "pickup", "holding": True})
    assert not done_already({"do": "pickup"}, None)
    bridge, body = _bridge(tmp_path)
    session = Session(Config(seed=7, nest=True))
    dispatch(session, "advance", {"until": "10:00"})
    at = len(session.core.log.events)
    dispatch(session, "act", {"action": "place_on_nest", "params": {}})
    bridge.after_call(session, at, done_by_hand={"hand": "place", "to": "nest"})
    _play(bridge)
    assert body.hands() == []
    flat_bridge, flat_body = _bridge(tmp_path)
    flat_body.home = "default_home,no_nest"
    flat_body.put(0.8, 0.3, 0.0)  # (picked up off the floor, away from its mat)
    duck = Session(Config(seed=7, nest=False))
    dispatch(duck, "advance", {"until": "10:00"})
    at = len(duck.core.log.events)
    dispatch(duck, "act", {"action": "pickup", "params": {}})
    flat_bridge.after_call(duck, at, done_by_hand={"hand": "pickup", "holding": True})
    _play(flat_bridge)
    assert [(hand["do"], hand.get("to")) for hand in flat_body.hands()] == [("place", "desk")]


def test_set_upright_is_the_worlds_hand_and_duck_core_is_not_told(tmp_path):
    """D2: 「扶起物理身體」 asks the world's hand; Duck Core's log does not change."""
    bridge, body = _bridge(tmp_path)
    answer = bridge.upright()
    assert answer["ok"] and body.hands() == [{"do": "upright"}]
    assert bridge.hand_note is not None and bridge.hand_note["do"] == "upright"
    bridge.ready = False
    with pytest.raises(ValueError):
        bridge.upright()


def test_a_fast_forward_places_the_body_at_the_plan_once_the_steps_stop(tmp_path):
    bridge, body = _bridge(tmp_path)
    bridge.sync_quiet_s = 0.4
    session = Session(Config(seed=7))
    dispatch(session, "advance", {"until": "10:00"})
    bridge.resync(session, "start")
    s = session.core.s
    for step in range(3):  # three fast-forward steps, 0.15 s apart
        s.pos_x, s.pos_y = 300 * (step + 1), 0
        bridge.skipped(session)
        time.sleep(0.15)
        assert _truth(bridge)["placed_why"] == "start"  # (still stepping: not yet)
    time.sleep(0.8)
    synced = _truth(bridge)
    assert synced["placed_why"] == "sync" and (synced["x_mm"], synced["y_mm"]) == (900, 0)
    places = [params for op, params in body.calls if op == "x_place"]
    assert len(places) == 2  # the start, and one sync for all three steps


def test_a_restarted_body_is_placed_again_where_the_duck_is(tmp_path):
    bridge, body = _bridge(tmp_path)
    session = Session(Config(seed=7))
    bridge.resync(session, "start")
    assert _truth(bridge)["placed_why"] == "start"
    snap = bridge.snapshot()
    assert snap["frame"]["truth"]["placed_why"] == "start" and snap["monitor"]["trust"] == "ok"
    assert snap["eye"]["error"] == "鴨眼畫面沒有開"  # (this bridge runs no camera)
    assert snap["world"]["physical"] and snap["world"]["frame"] == {
        "x_mm": 0,
        "y_mm": 0,
        "heading_mrad": 0,
    }
    bridge.restart()  # a new physics body (here it cannot start: no official checkout)
    assert not bridge.ready and bridge.snapshot()["frame"] is None
    fresh = StandInBody()
    fresh.put(2.0, 2.0, 1.0)  # a new body at its spawn
    bridge.body = fresh  # type: ignore[assignment]
    bridge.ready = True
    assert _truth(bridge)["placed_at"] is None  # (the old placement is not this body's)
    bridge.resync(session, "start")
    placed = _truth(bridge)
    plan = session.core.pose_now()
    assert (placed["x_mm"], placed["y_mm"]) == (round(plan[0] * 1000), round(plan[1] * 1000))


def test_the_homes_frame_moves_the_plan_into_the_world():
    """Duck Core's origin is the layout's nest (else its mat): identity in default_home; in
    the apartment the nest at (−3.50, −2.45) facing +y (owner decision B1)."""
    identity = HomeFrame()
    assert identity.to_world([1.2, -0.4, 0.5]) == pytest.approx((1.2, -0.4, 0.5))
    nest = HomeFrame(-3.5, -2.45, math.pi / 2)
    x, y, yaw = nest.to_world([0.45, 0.0, 0.0])  # "on the floor", just in front of the nest
    assert (x, y) == pytest.approx((-3.5, -2.0)) and yaw == pytest.approx(math.pi / 2)
    assert nest.to_plan((x, y, yaw)) == pytest.approx((0.45, 0.0, 0.0), abs=1e-9)
    from duckmind.world import homelayout

    home = homelayout.load("default_home")
    assert HomeFrame.of(home) == HomeFrame()
    assert HomeFrame.of(homelayout.load("default_home,no_nest")) == HomeFrame()


def test_the_heading_that_a_placement_keeps_survives_a_fall_on_the_face():
    for pitch_deg in (0.0, 10.0, 80.0, 90.0, 95.0, 100.0, 150.0):
        q = _quat(0.5, math.radians(pitch_deg))
        assert heading_of(q) == pytest.approx(0.5, abs=1e-9), pitch_deg
    assert yaw_of(_quat(0.5, math.radians(95.0))) == pytest.approx(0.5 - math.pi, abs=1e-9)
    leaning = [0.9937, 0.0436, 0.0436, 0.0998]  # ~5° of pitch and roll: the two agree
    assert heading_of(leaning) == pytest.approx(yaw_of(leaning), abs=math.radians(0.5))


def _quat(yaw: float, pitch: float = 0.0) -> list[float]:
    """Rz(yaw)·Ry(pitch), MuJoCo order (w, x, y, z)."""
    cz, sz, cy, sy = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return [cz * cy, -sz * sy, cz * sy, sz * cy]


def test_a_standing_body_is_placed_standing(tmp_path):
    """A body that stands when it is placed is set down standing (the keyframe of its running
    policy); Duck Core's posture follows by the usual posture request."""
    bridge, body = _bridge(tmp_path)
    body.policy = "walk"
    bridge.place((0.5, 0.5, 0.0), "sync")
    assert [params for op, params in body.calls if op == "x_place"][-1]["keyframe"] == "STAND"


def test_a_put_down_where_the_body_already_is_needs_no_hand(tmp_path):
    """蛋黃's desk is the mat on the floor (A3): 「放到地上」 from the desk moves nothing, so no
    hand comes for it; anywhere else (or tipped over, or held up) the hand does it."""
    bridge, body = _bridge(tmp_path)
    body.home = "default_home,no_nest"
    session = Session(Config(seed=7, nest=False))
    dispatch(session, "advance", {"until": "10:00"})
    at = len(session.core.log.events)
    dispatch(session, "set_setting", {"key": "floor", "value": True})  # at its mat: stays
    bridge.after_call(session, at)
    _play(bridge)
    assert body.hands() == [] and bridge.hand_note and bridge.hand_note.get("skipped")
    body.put(1.0, -0.5, 0.0)  # it walked off: back on the desk is a real carry
    at = len(session.core.log.events)
    dispatch(session, "set_setting", {"key": "floor", "value": False})
    bridge.after_call(session, at)
    _play(bridge)
    assert [(hand["do"], hand.get("to")) for hand in body.hands()] == [("place", "desk")]


def test_a_fall_only_the_plan_had_moves_no_hand(tmp_path):
    """Phase-2 review: Duck Core's family routine rights a fall its plan rolled (a dice roll);
    the physics body sits fine, so the world's hand does nothing for it. A body that really
    tipped over is righted as before."""
    bridge, body = _bridge(tmp_path)
    bridge._hand({"do": "upright"})
    assert body.hands() == [] and bridge.hand_note and bridge.hand_note.get("skipped")
    assert "沒有跌倒" in bridge.hand_note["why_zh"][0]
    tipped = body.frame()
    tipped["quat"] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]  # on its side
    body.frame = lambda: tipped  # type: ignore[method-assign]
    bridge._hand({"do": "upright"})
    assert body.hands() == [{"do": "upright"}]


def test_a_world_whose_camera_did_not_start_shows_no_picture_and_says_why(tmp_path):
    """Phase-2 review: when the world has a camera (x_sensors) that did not start, the page's
    window says so instead of phase 0's renderer drawing a scene the physics does not run."""
    from duckmind.simweb.live import NoEye

    bridge, body = _bridge(tmp_path)
    bridge.eye_enabled = True
    body.host_info = {"ops": ["x_place", "x_world", "x_hand", "x_people", "x_sensors"]}
    answers = {"x_sensors": {"media": {"socket": None}, "errors": ["感測器資料夾的路徑太長"]}}
    plain = body.world_call

    def world_call(op: str, **params: Any) -> dict[str, Any]:
        return answers[op] if op in answers else plain(op, **params)

    body.world_call = world_call  # type: ignore[method-assign]
    bridge._start_eye(body)  # type: ignore[arg-type]
    assert isinstance(bridge.eye, NoEye)
    status = bridge.eye_status()
    assert "鏡頭沒有啟動" in status["error"] and "路徑太長" in status["error_detail"]
    picture, tag, why = bridge.eye_image()
    assert picture is None and tag == "" and "鏡頭沒有啟動" in why


def test_the_hand_waits_while_a_demonstration_has_the_body(tmp_path):
    """Phase-2 review: a routine's hand (a pickup, into the nest) used to grab the body in the
    middle of a 示範; it now waits until the demonstration ends."""
    bridge, body = _bridge(tmp_path)
    bridge.demo_until = time.time() + 1.0
    with bridge._wake:
        heapq.heappush(bridge._due, (time.monotonic(), 0, {"hand": {"do": "pickup"}}))
    _play(bridge)
    assert body.hands() == []  # held back during the demonstration
    assert any("hand" in item[2] for item in bridge._due)
    bridge.demo_until = 0.0
    with bridge._wake:  # (its moment has come)
        bridge._due = [(time.monotonic(), order, item) for _, order, item in bridge._due]
        heapq.heapify(bridge._due)
    _play(bridge)
    assert body.hands() == [{"do": "pickup"}]


def test_a_body_still_starting_is_stopped_only_once_its_start_is_over(tmp_path):
    """Phase-2 measurement note: after 「重新孵化」 the old body's /private/tmp/dm-live-* folder
    was sometimes left behind — it was stopped while still starting (its processes and folder
    came after the stop), and simweb could exit before the stopping thread finished. The old
    body is now stopped once its start is over, and stop() waits for it."""
    bridge, body = _bridge(tmp_path)
    events: list[str] = []
    starting = threading.Thread(target=lambda: (time.sleep(0.4), events.append("started")))
    starting.start()
    bridge._thread = starting  # (the old body is still starting)
    body.stop = lambda: events.append("stopped")  # type: ignore[method-assign]
    bridge.restart()
    bridge.stop()  # (waits for the old body's stop)
    assert events == ["started", "stopped"]


def test_the_bridge_and_the_view_take_both_engines_ducks_and_their_falls(tmp_path):
    """Phase-4 landing (WP10): 蛋黃's old save opens on the frozen v4_0 engine and a new page
    duck is v4_1, so the live bridge gets either session. v4's fall is Duck Core's own
    ``body.fell`` (with where it lies and the walk it cut); v4_1's is the world's ``body.fell``
    percept followed by Duck Core's own ``duck.posture(fallen)`` carrying both. Either way the
    fall interrupts what was still waiting, the 3D view gets the same place to lay the body down
    (LANDING §3b) and the bridge places the body where the duck's plan is."""
    from phase4_harness import walking_age

    laid: dict[str, tuple[float, ...]] = {}
    for rev in ("v4_0", "v4_1"):
        config: dict[str, Any] = {"seed": 4, "name": "蛋黃", "fall_ppm_per_s": 1_000_000}
        if rev == "v4_1":
            config["engine_rev"] = "v4_1"
        session = open_session(config)
        assert session.core.__module__.startswith(
            "duckmind.engines.v4_0" if rev == "v4_0" else "duckmind.duckcore"
        )
        walking_age(session)
        bridge, body = _bridge(tmp_path)
        bridge.resync(session, "start")
        events = session.core.log.events
        before = len(events)
        for _ in range(12):  # (every walk falls at this rate: the first outing ends in one)
            dispatch(session, "advance", {"minutes": 10})
            if any(event.kind == "body.fell" for event in events[before:]):
                break
        index = next(i for i in range(before, len(events)) if events[i].kind == "body.fell")
        assert events[index].kind in INTERRUPTS  # the bridge drops what was still waiting
        moves = [_motion(event) for event in events[index : index + 3]]
        lying = next(
            move for move in moves if move and move.get("at") is not None
            and (move.get("fall") or move.get("posture") == "fallen")
        )  # fmt: skip
        laid[rev] = tuple(lying["at"])
        assert lying.get("cut") is not None  # the walk it cut, where it stopped
        bridge.after_call(session, before)  # (either engine's events: no error)
        _play(bridge)
        bridge.resync(session, "sync")
        placed = _truth(bridge)
        plan = session.core.pose_now()
        assert (placed["x_mm"], placed["y_mm"]) == (round(plan[0] * 1000), round(plan[1] * 1000))
    assert laid["v4_0"] == laid["v4_1"]  # the same duck, the same fall, the same place


@needs_node
def test_beak_provenance_survives_unpack_interpolation_and_frame_pose():
    out = _node(
        """
const names = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
const frame = (t, w, source, m = 0.25) => ({
  t, w, j: [0, 0, 0, 0], p: [0, 0, 0.1], q: [1, 0, 0, 0], m, m_source: source,
});
const sourceAt = (left, right = left) => {
  const buffer = T.mergeFrames([], [frame(1, 10, left), frame(1.1, 10.1, right)].map((f) => T.unpackFrame(f, names)));
  const sample = T.sampleFrames(buffer, 10.05).pose;
  return T.framePose(sample).mouthSource;
};
const unpacked = T.unpackFrame(frame(1, 10, "measured_sim"), names);
const noMouth = T.framePose(T.unpackFrame(frame(1, 10, "measured_sim", null), names)).mouthSource;
const i18n = { t: (key, params) => ({ key, params: params || null }) };
console.log(JSON.stringify({
  unpacked: unpacked.mouthSource,
  measured: sourceAt("measured_sim"),
  commanded: sourceAt("commanded_fallback"),
  mixed: sourceAt("measured_sim", "commanded_fallback"),
  unknown: sourceAt("measured_sim", "unavailable"),
  missing: sourceAt("measured_sim", null),
  missingMouth: noMouth,
  captions: [
    T.beakCaption(i18n, "measured_sim", 3),
    T.beakCaption(i18n, "commanded_fallback", 7),
    T.beakCaption(i18n, "unavailable", null),
  ],
}));
"""
    )
    assert out == {
        "unpacked": "measured_sim",
        "measured": "measured_sim",
        "commanded": "commanded_fallback",
        "mixed": "commanded_fallback",
        "unknown": "unavailable",
        "missing": "unavailable",
        "missingMouth": "unavailable",
        "captions": [
            {"key": "page.duck3d.readout.beak.measured", "params": {"value": 3}},
            {"key": "page.duck3d.readout.beak.commanded", "params": {"value": 7}},
            {"key": "page.duck3d.readout.beak.unavailable", "params": None},
        ],
    }
