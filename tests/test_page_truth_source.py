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

Static checks read the page's code; the behavioural ones run ``static/truth.js`` in node and the
server's bridge with a stand-in body that answers the world ops (no robotd, no MuJoCo).
"""

from __future__ import annotations

import heapq
import json
import math
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from duckmind.duckcore.api import Config, Session, _motion, dispatch, open_session
from duckmind.livebody import LivePaths
from duckmind.simweb import live
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
DUCK3D_JS = STATIC / "duck3d.js"


def _zh_words() -> dict[str, str]:
    """The page's 繁體中文 words by catalog key (static/i18n.zh-Hant.*.js; docs/i18n.md)."""
    words: dict[str, str] = {}
    for path in sorted(STATIC.glob("i18n.zh-Hant.*.js")):
        body = path.read_text(encoding="utf-8")
        words.update(
            json.loads(body[body.index("{", body.index("DuckI18n.load(")) : body.rindex("}") + 1])
        )
    return words


def _worded(text: str) -> str:
    """A page source as its words were written before the page spoke six languages: every
    quoted catalog key replaced by its 繁體中文 words, the data-i18n markup left out."""
    words = _zh_words()
    text = re.sub(r'\s+data-i18n(?:-html|-attr)?="[^"]*"', "", text)
    return re.sub(
        r"([\"'])((?:page|fmt|web)\.[A-Za-z0-9_.|<>-]+)\1",
        lambda m: m.group(1) + words.get(m.group(2), m.group(2)) + m.group(1),
        text,
    )


HOME3D_JS = STATIC / "home3d.js"


def _function(source: str, name: str) -> str:
    """The text of one top-level ``function name(...) {...}`` in a script."""
    start = source.index(f"function {name}(")
    depth = 0
    for index in range(source.index("{", start), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function {name}")


# ── static: the 3D view draws the solid duck from the truth only ─────────────────


def test_the_solid_duck_is_composed_from_the_truth_and_the_plan_moves_only_the_shadow():
    duck3d = DUCK3D_JS.read_text()
    assert 'from "./truth.js"' in duck3d and 'from "./home3d.js"' in duck3d
    step = _function(duck3d, "step")
    # The frame's truth (and only the frame) decides the solid duck; one applyPose, of it.
    assert "truthTarget(view.external)" in step
    assert "compose({ plan, truth: view.live.shown" in step
    assert step.count("applyPose(") == 1 and "applyPose(drawn.solid)" in step
    assert "applyShadow(drawn.shadow" in step
    # Duck Core's position appears once, moved into the world by the home's frame, and goes
    # into the plan it hands over (the shadow's pose).
    move = "planToWorld(view.room.frame, view.body.x, view.body.y, view.body.yaw)"
    assert step.count(move) == 1
    plan = step[step.index("const plan = {") : step.index("};", step.index("const plan = {"))]
    assert "[planX, planY," in plan and "yawRollQuat(planYaw," in plan
    for field in ("x", "y", "yaw"):
        # (once in the move into the world; x and y once more for the readout's distance)
        found = len(re.findall(rf"\bview\.body\.{field}\b", step))
        assert found == (1 if field == "yaw" else 2), field
        assert f"view.body.{field}" not in plan, field
    # Gone: the physics lean put on Duck Core's heading and position (the old hybrid), the
    # demonstration drawn relative to Duck Core's pose, and Duck Core's fall drawn instead
    # of the physics body.
    for old in ("withHeading(", "demoPose(", "ownBody", "smoothExternal(", "extMix"):
        assert old not in duck3d, old
    # The whole frame reaches the view (truth included), never Duck Core's snapshot position.
    poll = _function(duck3d, "pollLive")
    assert "setExternalPose(info.frame)" in poll
    assert "position" not in _function((TRUTH_JS).read_text(), "truthTarget")


def test_the_server_computes_the_truth_from_the_frame_only():
    """G15: the truth is the frame, in world coordinates — never Duck Core's pose. Duck Core's
    pose (pose_now) is only ever where a world operation puts the body (x_place: a body that
    starts, a sync after a fast-forward)."""
    import ast
    import inspect
    import textwrap

    def code(function: Any) -> str:  # (without its docstring and comments)
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        body = tree.body[0]
        assert isinstance(body, ast.FunctionDef)
        if isinstance(body.body[0], ast.Expr) and isinstance(body.body[0].value, ast.Constant):
            body.body = body.body[1:]
        return ast.unparse(body)

    truth = code(LiveBridge.truth)
    for plan in ("pose_now", "session", "core", "pos_x", ".s.", "frame.to_world", "_anchor"):
        assert plan not in truth, plan
    source = inspect.getsource(live)
    uses = [line.strip() for line in source.splitlines() if "pose_now()" in line]
    # (only LiveBridge.plan_pose reads it: where a placement puts the body — the plan's pose,
    # or the physical dock for a duck that sits in its nest; resync and skipped call it)
    assert uses == ["return self.frame.to_world(core.pose_now())"]
    assert "pose_now" in code(LiveBridge.plan_pose)
    assert "self.plan_pose(session)" in code(LiveBridge.resync)
    assert "self.plan_pose(session)" in code(LiveBridge.skipped)
    # Phase 0's re-seating anchors and the interim hand anchor are gone.
    for old in ("def anchor(", "anchor_pose", "hand_pose", "_room_of", "HAND_SETTLE_S", "_settle"):
        assert old not in source, old


def test_the_page_says_what_the_physics_has_and_which_duck_it_draws():
    duck3d = _worded(DUCK3D_JS.read_text())
    truth = _worded(TRUTH_JS.read_text())
    assert "物理真相" in truth and "這是小鴨的計畫，不是物理" in truth
    # The home is solid (it is in the physics); translucent only while the physics runs
    # another world, and then the words say so (W07).
    assert "房子、家具和玩具在物理裡都是真的" in truth
    assert "只有畫面，物理裡沒有" in truth
    step = _function(duck3d, "step")
    assert "setGhost(home, truthMode && !view.room.physical)" in step
    assert "physicalHome: view.room.physical" in _function(duck3d, "renderMode")
    # W05: when the plan jumps (walking "into" the nest) the page says the body did not.
    assert "不會瞬移" in duck3d and "PLACED_ZH" in _function(duck3d, "play")
    assert "motion.place.length < 3" in _function(duck3d, "play")
    assert "物理裡還沒有手" not in duck3d  # (phase 2: there is a hand)
    placed = duck3d[
        duck3d.index("const PLACED_WHY_ZH = {") : duck3d.index(
            "};", duck3d.index("const PLACED_WHY_ZH = {")
        )
    ]
    assert "start:" in placed and "sync:" in placed and "hand:" not in placed
    hands = duck3d[
        duck3d.index("const HAND_ZH = {") : duck3d.index("};", duck3d.index("const HAND_ZH = {"))
    ]
    for do in ("pickup:", "place:", "upright:"):
        assert do in hands, do
    # The room comes from the home the physics runs (home3d.js); nothing is hard-coded any more.
    assert "function buildRoom(" not in duck3d and "furniture(scene" not in duck3d
    assert "buildHome(view.scene, home" in _function(duck3d, "loadHome")
    assert '"/api/home"' in _function(duck3d, "loadHome")
    assert "/api/body?home=" in _function(duck3d, "pollLive")
    # The lights are the server's (world/lights.py); the page's own rules only for an older server.
    light = _function(duck3d, "lightTarget")
    assert light.index("view.serverLight") < light.index("sun.rise")
    assert 'addEventListener("duck:lights"' in duck3d


def test_the_home_is_drawn_from_the_layout_and_walls_in_the_way_fade():
    """B4 and the physical room: home3d.js draws every element of the layout (page JSON: box,
    cylinder, frustum, sphere, ellipsoid, capsule, plane), the toys, the hand, the family's
    bodies and the nest's real AprilTag cells; walls between the camera and the duck fade."""
    home = HOME3D_JS.read_text()
    for shape in ('"plane"', '"cylinder"', '"frustum"', '"sphere"', '"ellipsoid"', '"capsule"'):
        assert f"case {shape}" in home, shape
    build = _function(home, "buildHome")
    assert "el.flags.page === false" in build  # camera-only bands and ceilings are not drawn
    for part in ("payload.hand", "payload.people", "payload.nest_parts", "layout.things"):
        assert part in build, part
    fade = _function(home, "fadeWalls")
    assert "intersectBox" in fade and "FADED_OPACITY" in fade
    assert 'new Set(["wall", "railing", "gate"])' in home
    # The hand held by the weld is drawn on the drawn duck's trunk (no lag behind the body).
    assert "home.handHolding && duck && duck.trunk" in _function(home, "stepWorld")
    assert "tagTexture(tag.cells" in home  # the same bits as the physics' texture
    step = _function(DUCK3D_JS.read_text(), "step")
    assert "fadeWalls(home, view.camera" in step and "stepWorld(home, dt" in step


def test_the_duck_eye_window_and_the_trust_badge_work_in_3d_and_2d():
    page = (STATIC / "index.html").read_text()
    area = page[page.index('<div class="stage-area">') : page.index('<div class="duck3d-readout"')]
    three = area[area.index('id="duck3d"') : area.index('id="duckStage"')]
    for hook in ("stageMode", "trustBadge", "trustWhy", "eyeWindow", "btnEye", "eyeImg", "eyeNote"):
        assert f'id="{hook}"' in area, hook  # in the stage area…
        assert f'id="{hook}"' not in three, hook  # …not inside #duck3d (gone in 2D)
    assert "牠的鏡頭看到的" in page and "每 2 秒一張" in page and "物理不可信" in page
    script = (STATIC / "app.js").read_text()
    assert "EYE_EVERY_MS = 2000" in script and '"/api/eye.jpg"' in script
    wanted = _function(script, "eyeWanted")
    for gate in ("eyeUi.open", "eyeUi.inView", "document.hidden", "eyeFast()"):
        assert gate in wanted, gate
    # F4: the world's camera gives a frame only while the window asks (asking opens it).
    assert "eye.window" in wanted
    assert "IntersectionObserver" in script and '"visibilitychange", eyeSchedule' in script
    # In 2D (no 3D view polling /api/body) the page asks for the body itself.
    assert "fetch(`/api/body?home=" in _function(script, "askBody")


def test_the_phase_2_controls_are_on_the_page_and_wired():
    """D2 「扶起物理身體」 next to the fallback 「重設物理身體」; E1 the world battery beside Duck
    Core's; the physical home's line; card 09 (示範 bump/peck/kick, roll a ball, tidy up,
    set upright) inside [data-scenarios], handled by app.js's delegated clicks."""
    page = (STATIC / "index.html").read_text()
    for hook in ('id="liveUpright"', 'id="liveReset"', 'id="worldBattery"', 'id="worldLine"'):
        assert hook in page, hook
    lab = page[page.index('id="labTools"') : page.index("</section>", page.index('id="labTools"'))]
    for button in (
        'data-world="body_show" data-kind="bump"',
        'data-world="body_show" data-kind="peck"',
        'data-world="body_show" data-kind="kick"',
        'data-world="body_roll"',
        'data-world="body_tidy"',
        'data-world="body_upright"',
    ):
        assert button in lab, button
    assert '<span class="num">09</span>' in lab and 'id="worldThing"' in lab
    script = (STATIC / "app.js").read_text()
    click = _function(script, "onScenarioClick")
    assert "button.dataset.world" in click and "safe(button.dataset.world, params)" in click
    assert "result.deferred" in click  # a family member walks over first (D3)
    assert "params, visit: true" in click  # (only these buttons: a banner's steps stay in order)
    world = _function(script, "renderWorld")
    assert "#worldBattery" in world and "#worldLine" in world and "takeHome(" in world
    # The toy chooser is filled from the home: /api/body's home_json (2D) and the home the 3D
    # view drew — the view fetches /api/home itself, so /api/body never carries home_json
    # there (a chooser left with only the page's one placeholder toy, found by the gate run).
    assert "#worldThing" in _function(script, "takeHome")
    assert 'addEventListener("duck:home"' in script
    duck3d = DUCK3D_JS.read_text()
    assert 'CustomEvent("duck:home"' in _function(duck3d, "loadHome")
    assert 'getElementById("liveUpright")' in duck3d and '"body_upright"' in duck3d


def test_the_new_page_styles_use_only_the_design_tokens():
    css = (STATIC / "style.css").read_text()
    block = css[css.index("/* Live physics overlays") : css.index(".duck3d-fx {")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)  # colours only as tokens
    assert "border-radius" not in block and "blur(" not in block
    for token in ("--truth:", "--plan:"):
        assert token in css
    assert "var(--truth)" in block and "var(--plan)" in block
    world = css[css.index("/* The physical home (phase 2)") : css.index(".world-tools select")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", world)
    assert "border-radius" not in world and "blur(" not in world
    # (a colour written out as rgba()/hsl() is a colour outside the tokens too)
    for part in (block, world):
        assert not re.search(r"\b(rgba?|hsla?)\(", part)


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


def test_a_broken_body_raises_the_warning_and_the_2d_duck_is_labelled_the_plan():
    """Owner decision (2): when the physics is abnormal the page says 「物理不可信」 and why —
    a dead robotd or body server included. The 2D duck (no 3D view) only plays the plan."""
    script = (STATIC / "app.js").read_text()
    trust = _function(script, "renderTrust")
    assert "info.error" in trust and "error_zh" in trust
    assert "info.available &&" not in trust  # (it used to hide the badge exactly then)
    flat = _function(script, "renderMode2d")
    assert 'contains("has3d")' in flat and "#stageMode" in flat and "PLAN_TITLE" in flat
    assert "renderMode2d(info)" in _function(script, "takeBody")
    title = re.search(r'const PLAN_TITLE = "([^"]+)";', script)
    same = re.search(r'export const PLAN_TITLE = "([^"]+)";', TRUTH_JS.read_text())
    assert title and same and title.group(1) == same.group(1)
    duck3d = DUCK3D_JS.read_text()
    assert "broken: !!info.error" in _function(duck3d, "renderMode")
    assert "error_zh" in _function(duck3d, "liveText")


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


def test_the_view_lays_a_v4_1_fall_down_where_duck_core_says():
    """LANDING §3b: v4 put where the duck fell on its ``body.fell``; v4_1 puts it on Duck Core's
    own ``duck.posture(fallen)``, which ``api._motion`` hands the page as ``at``. The 3D view lays
    the fallen body there either way — when the fall is played (``nextBodyAction``) and when it is
    applied at once after a fast-forward (``settle``) — and a posture item carries ``at`` along
    (``play``, ``queueItem``), so the plan duck does not wait for the next snapshot to lie where
    it fell."""
    source = DUCK3D_JS.read_text(encoding="utf-8")
    settle = _function(source, "settle")
    assert 'if (item.posture === "fallen") placeAt(b, item.at);' in settle
    assert 'if (item.fall || item.posture === "fallen") placeAt(b, item.at);' in _function(
        source, "nextBodyAction"
    )
    assert "{ posture: motion.posture, at: motion.at, caption: text }" in _function(
        source, "queueItem"
    )
    assert "b.queue.push({ posture: motion.posture, at: motion.at, caption: motion.caption })" in (
        _function(source, "play")
    )


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


def test_truth_beak_caption_uses_provenance_and_no_longer_claims_no_jaw():
    duck3d = DUCK3D_JS.read_text()
    assert "beakCaption(window.I18N, source, beakValue)" in duck3d
    assert "模擬身體沒有嘴巴關節" not in duck3d
