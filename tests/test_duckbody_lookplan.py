"""lookplan (spec §4.2 rules, look_veto, owner A2's head-down watch): pure, integer, no clock."""

from __future__ import annotations

from typing import Any

import pytest

from duckmind.duckbody import lookplan as L
from duckmind.duckbody.protocol import LookRequest
from duckmind.duckcore import content
from duckmind.expression_sweep import looks


def _offsets(pose: dict[str, float], side: int = 1) -> L.Offsets:
    targets = content.head_targets(pose, side)
    names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    vals = [round((targets[n] - content.HEAD_HOME[n]) * 1000) for n in names]
    return (vals[0], vals[1], vals[2], vals[3])


def test_catalogue_poses_are_the_validated_clips() -> None:
    assert _offsets(looks.HALF_DOWN) == L.HALF_DOWN
    assert _offsets(looks.UP) == L.UP
    assert _offsets({"hy": looks.AIM})[2] == L.AIM_YAW_MRAD
    around = dict(content.CLIPS["look_around"])
    assert _offsets(around[700])[2] == L.SCAN_LEFT_MRAD
    assert _offsets(around[2300])[2] == L.SCAN_RIGHT_MRAD
    assert L.HEAD_DOWN_WALK == L.HALF_DOWN  # the A100's head-down walking pose
    for spec in L.CATALOGUE.values():
        for key in spec.keys:
            assert abs(key.pose[2]) <= L.AIM_YAW_MRAD
            assert all(isinstance(v, int) for v in key.pose)
    assert L.CATALOGUE["scan"].captures == 3
    assert L.CATALOGUE["glance_down"].safe and L.CATALOGUE["end_look"].safe


def test_rules_one_and_three() -> None:
    assert L.legs_quiet(1000, None)
    assert not L.legs_quiet(1000, 700)
    assert L.legs_quiet(1150, 700)
    settle = L.HomeSettle()
    assert not settle.step(0, True)
    assert not settle.step(200, True)
    assert settle.step(300, True)
    assert not settle.step(310, False)
    assert not settle.step(320, True)


def test_head_at_tolerance() -> None:
    home = (0.349, 0.349, 0.0, 0.0)
    assert L.head_at((0.349, 0.349, 0.0, 0.0), home, L.HOME)
    assert not L.head_at((0.349 - 0.125, 0.349 + 0.175, 0.0, 0.0), home, L.HOME)
    assert L.head_at((0.349 - 0.125, 0.349 + 0.175, 0.0, 0.0), home, L.HALF_DOWN)


def test_need_glance_triggers() -> None:
    base = dict(
        now_ms=100_000,
        heading_mrad=0,
        last_glance_ms=90_000,
        last_glance_heading_mrad=0,
        light="bright",
    )
    assert not L.need_glance(**base)
    assert L.need_glance(**{**base, "first_leg": True})
    assert L.need_glance(**{**base, "after_arc": True})
    assert L.need_glance(**{**base, "set_down": True})
    assert L.need_glance(**{**base, "after_contact": True})
    assert L.need_glance(**{**base, "after_unseen": True})  # one step, one look
    assert L.need_glance(**{**base, "last_glance_ms": 30_000})  # > 60 s ago
    assert L.need_glance(**{**base, "heading_mrad": 400})  # turned > 20°
    assert not L.need_glance(**{**base, "heading_mrad": 300})
    assert not L.need_glance(**{**base, "first_leg": True, "light": "dark"})  # cannot see


def test_need_glance_follows_the_callers_attention() -> None:
    """Owner 2026-09-28 ~12:30 「保留鴨子偶爾不注意撞到東西的呆萌感」: whether the duck glances
    before a walk is the caller's attention, never chance. ``on`` (the default) is every rule;
    ``low`` only what catches anyone's eye; ``elsewhere`` never."""
    base = dict(
        now_ms=100_000,
        heading_mrad=0,
        last_glance_ms=90_000,
        last_glance_heading_mrad=0,
        light="bright",
    )
    triggers = {
        "first_leg": True,
        "set_down": True,
        "after_contact": True,
        "after_arc": True,
        "after_unseen": True,
        "last_glance_ms": 1_000,
        "heading_mrad": 900,
    }
    strong = {"first_leg", "set_down", "after_contact"}
    for name, value in triggers.items():
        case = {**base, name: value}
        assert L.need_glance(**case) and L.need_glance(**case, attention="on")
        assert L.need_glance(**case, attention="low") is (name in strong)
        assert not L.need_glance(**case, attention="elsewhere")
    assert not L.need_glance(**base, attention="low")
    # an unknown light: an attentive duck glances on a rule (the glance is how it is learnt),
    # a duck of low attention does not
    assert L.need_glance(**{**base, "light": "none", "after_arc": True})
    assert not L.need_glance(**{**base, "light": "none", "after_arc": True}, attention="low")


def test_motion_model_and_route() -> None:
    assert L.speed_mm_s(400) == 195
    assert L.speed_mm_s(350) == 178
    assert L.speed_mm_s(550) == 317
    assert L.speed_mm_s(475) == 227
    speeds = [L.speed_mm_s(v) for v in range(300, 560, 10)]
    assert speeds == sorted(speeds)
    route = L.predict_route(L.Pose2(0, 0, 0), 400, 0, 3000)
    last_real = route[-2]
    assert L.distance_mm(last_real, L.Pose2(0, 0, 0)) == pytest.approx(
        585, abs=15
    )  # 0.195 m/s × 3 s
    assert last_real.y_mm < -100  # the measured right drift (−10 °/s: Duck Core trims it)
    assert L.distance_mm(route[-1], last_real) == pytest.approx(
        L.ROUTE_TAIL_MM + L.ROUTE_MARGIN_MM, abs=3
    )
    for p in route:
        assert all(isinstance(v, int) for v in (p.x_mm, p.y_mm, p.h_mrad))
    arc = L.predict_route(L.Pose2(0, 0, 0), 400, -1000, 2000)
    assert arc[-2].h_mrad < -700  # right arc: ~−44 °/s


TRIM = 293  # the vyaw that cancels the measured right drift (bias −175 mrad/s, gain 0.597)


def _seen_corridor(grid: L.VetoGrid, now: int, level: str, length: int = 1500) -> None:
    # one look straight ahead: five sectors, all free to ``length``
    grid.mark_view(
        L.Pose2(0, 0, 0), [(b, 70, length) for b in (-260, -130, 0, 130, 260)], now, level
    )
    grid.mark_walked([L.Pose2(0, 0, 0), L.Pose2(100, 0, 0), L.Pose2(200, 0, 0)], now, level)


UNSEEN_STEP = L.Veto(True, L.UNSEEN_STEP_MS, "unseen", L.UNSEEN_VX_MM)


def test_look_veto_clear_shortened_unseen() -> None:
    grid = L.VetoGrid()
    # owner 2026-09-28 「3. 一律放寬」: nothing seen is no longer a refusal in the light — one
    # short step at the approach speed into the unseen floor
    assert L.look_veto(grid, L.Pose2(0, 0, 0), 550, TRIM, 3000, 1000, "bright") == UNSEEN_STEP
    assert L.look_veto(grid, L.Pose2(0, 0, 0), 400, TRIM, 1200, 1000, "dim") == L.Veto(
        True, 1200, "unseen", L.UNSEEN_VX_MM
    )  # never longer than the walk asked for
    _seen_corridor(grid, 1000, "bright", length=1500)
    clear = L.look_veto(grid, L.Pose2(0, 0, 0), 400, TRIM, 3000, 2000, "bright")
    assert clear.ok and clear.max_ms == 3000 and clear.why == "clear" and clear.vx_cap_mm is None
    longer = L.look_veto(grid, L.Pose2(0, 0, 0), 400, TRIM, 8000, 2000, "bright")
    assert longer.ok and longer.why == "shortened" and 2000 <= longer.max_ms < 8000
    assert longer.vx_cap_mm is None  # on seen floor the walk keeps its own speed
    # untrimmed, the walk drifts ~30° right in 3 s and leaves the cone that was seen
    drift = L.look_veto(grid, L.Pose2(0, 0, 0), 400, 0, 3000, 2000, "bright")
    assert drift.ok and drift.why == "shortened" and drift.max_ms < 3000
    # stale cells are unseen again (a short step, not a refusal); dim frames stay fresh 5 min
    assert L.look_veto(grid, L.Pose2(0, 0, 0), 400, TRIM, 3000, 1000 + 600_001, "bright") == (
        UNSEEN_STEP
    )
    dim = L.VetoGrid()
    _seen_corridor(dim, 1000, "dim")
    assert L.look_veto(dim, L.Pose2(0, 0, 0), 400, TRIM, 3000, 1000 + 299_000, "dim").why == (
        "clear"
    )
    assert L.look_veto(dim, L.Pose2(0, 0, 0), 400, TRIM, 3000, 1000 + 301_000, "dim").why == (
        "unseen"
    )


def test_bright_cells_are_trusted_ten_minutes_not_five() -> None:
    """Owner 2026-09-28 「2. 10分鐘」: a cell seen free in bright stays fresh for 10 min (not the
    5 min once proposed); a dim one for 5 min."""
    assert L.FRESH_MS == {"bright": 600_000, "dim": 300_000}
    grid = L.VetoGrid()
    grid.mark(500, 0, 1000, "bright")
    for later in (300_001, 450_000, 599_999, 600_000):
        assert grid.fresh(500, 0, 1000 + later, "bright"), later
        assert grid.fresh(500, 0, 1000 + later, "dim"), later
    assert not grid.fresh(500, 0, 1000 + 600_001, "bright")
    dim = L.VetoGrid()
    dim.mark(500, 0, 1000, "dim")
    assert dim.fresh(500, 0, 1000 + 300_000, "bright")
    assert not dim.fresh(500, 0, 1000 + 300_001, "bright")
    # a route seen in bright 9 minutes ago is walked in full
    seen = L.VetoGrid()
    _seen_corridor(seen, 1000, "bright", length=1500)
    assert L.look_veto(seen, L.Pose2(0, 0, 0), 400, TRIM, 3000, 1000 + 540_000, "bright").why == (
        "clear"
    )


def test_an_unlearned_eye_never_stops_the_walk() -> None:
    """Phase-3 fix open item (owner 「3. 一律放寬」): an eye that has not learned its floor claims
    at most 350 mm of free floor (duck-eye's BOOT_CAP_MM); the veto wanted the route plus 220 mm
    seen free past 300 mm, so every walk ended ``veto_look`` and the floor was never learned. Now
    the walk steps into the unseen floor — from every heading, at every speed, in bright and dim."""
    for level in ("bright", "dim"):
        for vyaw in (-1000, -500, 0, TRIM, 500, 1000):
            for vx in (350, 400, 550):
                grid = L.VetoGrid()
                boot = 350 if level == "bright" else 280  # dim: the eye takes 20 % off
                grid.mark_view(
                    L.Pose2(0, 0, 0), [(b, 70, boot) for b in (-260, -130, 0, 130, 260)], 0, level
                )
                veto = L.look_veto(grid, L.Pose2(0, 0, 0), vx, vyaw, 3000, 100, level)
                assert veto.ok and veto.max_ms >= L.MIN_SEGMENT_MS, (level, vyaw, vx, veto)
                assert veto.vx_cap_mm in (None, L.UNSEEN_VX_MM)


def test_what_a_look_saw_end_the_floor_still_counts() -> None:
    """The relaxation is for floor nobody has seen: where a recent look saw the floor end at a
    wall or a thing, the step stops before it, and with no room at all the walk is refused
    (``seen_blocked``: turn another way). An ``unknown`` end — every end of an eye that has not
    learned its floor — blocks nothing; a newer look that sees the cells free clears them."""
    ahead = [(b, 70, 700, "wall") for b in (-260, -130, 0, 130, 260)]
    grid = L.VetoGrid()
    grid.mark_view(L.Pose2(0, 0, 0), ahead, 1000, "bright")
    assert grid.blocked(800, 0, 2000) and not grid.fresh(800, 0, 2000, "bright")
    assert not grid.blocked(800, 0, 1000 + 600_001)  # a blocked cell goes stale like a free one
    walk = L.look_veto(grid, L.Pose2(0, 0, 0), 400, TRIM, 6000, 2000, "bright")
    assert walk.ok and walk.why == "shortened" and walk.max_ms < 3000  # stops before the wall
    near = L.look_veto(grid, L.Pose2(450, 0, 0), 400, TRIM, 3000, 2000, "bright")
    assert near == L.Veto(False, 0, "seen_blocked")  # the wall is right there
    relaxed = L.look_veto(
        grid, L.Pose2(450, 0, 0), 400, TRIM, 3000, 2000, "bright", seen_blocks=False
    )
    assert relaxed.ok and relaxed.why == "unseen"  # (the switch the owner could still flip)
    unknown = L.VetoGrid()
    unknown.mark_view(
        L.Pose2(0, 0, 0), [(b, 70, 350, "unknown") for b in (-260, 0, 260)], 0, "bright"
    )
    assert not unknown.blocked(500, 0, 100)
    boot = L.look_veto(unknown, L.Pose2(0, 0, 0), 400, TRIM, 3000, 100, "bright")
    assert boot.ok and boot.why in ("unseen", "shortened")
    # an unseen step stops short of a thing seen further on
    far = L.VetoGrid()
    far.mark_view(L.Pose2(0, 0, 0), [(b, 70, 300, "unknown") for b in (-260, 0, 260)], 0, "bright")
    far.mark_view(L.Pose2(0, 0, 0), [(b, 70, 520, "thing") for b in (-40, 0, 40)], 0, "bright")
    step = L.look_veto(far, L.Pose2(0, 0, 0), 400, TRIM, 3000, 100, "bright")
    assert step.ok and step.why == "unseen" and L.MIN_SEGMENT_MS <= step.max_ms < L.UNSEEN_STEP_MS
    # the thing was taken away: a newer look sees the floor go on
    grid.mark_view(
        L.Pose2(0, 0, 0), [(b, 70, 1500, "unknown") for b in (-260, 0, 260)], 3000, "bright"
    )
    assert not grid.blocked(800, 0, 3000) and grid.fresh(800, 0, 3000, "bright")
    marks = L.view_marks(
        {"results": {"floor": {"free_mm": [300, 400, 500, 600, 700], "ends": ["wall", "thing"]}}}
    )
    assert [m[3] for m in marks] == ["wall", "thing", "unknown", "unknown", "unknown"]


def test_look_veto_in_the_dark_refuses_nothing_and_caps_nothing() -> None:
    """Owner 2026-09-28 ~12:30 「2. 放寬」 (C2 reversed): the dark refuses no walk, and since the
    phase-4 landing (「一律放寬」) it no longer caps a segment at 3 s either. Nothing seen → the
    slow unseen step, as in the light; what was seen free while the eye could see counts for the
    same 10 / 5 min whatever the light is now; every answer is the light's own."""
    for light in ("dark", "black", "none"):
        v = L.look_veto(L.VetoGrid(), L.Pose2(0, 0, 0), 550, TRIM, 3000, 1000, light)
        assert v == UNSEEN_STEP, light
    assert not hasattr(L, "DARK_SEGMENT_MS")
    dim = L.VetoGrid()
    _seen_corridor(dim, 1000, "dim", length=2500)
    in_dim = L.look_veto(dim, L.Pose2(0, 0, 0), 400, TRIM, 6000, 2000, "dark")
    assert in_dim.ok and in_dim.max_ms > 3000 and in_dim.why in ("clear", "shortened")
    assert in_dim == L.look_veto(dim, L.Pose2(0, 0, 0), 400, TRIM, 6000, 2000, "bright")
    stale_dim = L.look_veto(dim, L.Pose2(0, 0, 0), 400, TRIM, 6000, 1000 + 300_001, "dark")
    assert stale_dim == L.Veto(True, L.UNSEEN_STEP_MS, "unseen", L.UNSEEN_VX_MM)
    bright = L.VetoGrid()
    _seen_corridor(bright, 1000, "bright", length=2500)
    dark = L.look_veto(bright, L.Pose2(0, 0, 0), 400, TRIM, 6000, 2000, "dark")
    assert dark.ok and dark.max_ms > 3000 and dark.why in ("clear", "shortened")
    assert dark == L.look_veto(bright, L.Pose2(0, 0, 0), 400, TRIM, 6000, 2000, "bright")
    still = L.look_veto(bright, L.Pose2(0, 0, 0), 400, TRIM, 6000, 1000 + 600_000, "black")
    assert still == dark  # 10 min, the same as in the light
    later = L.look_veto(bright, L.Pose2(0, 0, 0), 400, TRIM, 6000, 1000 + 600_001, "black")
    assert later.ok and later.why == "unseen" and later.vx_cap_mm == L.UNSEEN_VX_MM
    assert L.look_veto(bright, L.Pose2(0, 0, 0), 400, TRIM, 6000, 2000, "bright").why == "clear"
    # a wall a lit look saw right there still turns the walk away in the dark
    wall = L.VetoGrid()
    wall.mark_view(L.Pose2(0, 0, 0), [(b, 70, 200, "wall") for b in (-260, 0, 260)], 0, "bright")
    assert L.look_veto(wall, L.Pose2(0, 0, 0), 400, TRIM, 3000, 100, "black") == L.Veto(
        False, 0, "seen_blocked"
    )


def test_an_inattentive_walk_goes_as_asked_up_to_what_it_saw() -> None:
    """Owner 2026-09-28 ~12:30 「呆萌感」: ``careful=False`` (attention low or elsewhere) — no
    slow unseen step, no stop where the seen floor ends; only what a recent look saw in the way
    stops it."""
    start = L.Pose2(0, 0, 0)
    none = L.look_veto(L.VetoGrid(), start, 550, TRIM, 3000, 1000, "bright", careful=False)
    assert none == L.Veto(True, 3000, "unseen")  # its own speed, the whole segment
    dark = L.look_veto(L.VetoGrid(), start, 550, TRIM, 6000, 1000, "black", careful=False)
    assert dark == L.Veto(True, 6000, "unseen")  # (no 3 s dark segment since the landing)
    part = L.VetoGrid()
    part.mark_view(start, [(b, 70, 700) for b in (-260, -130, 0, 130, 260)], 0, "bright")
    careful = L.look_veto(part, start, 400, TRIM, 6000, 100, "bright")
    careless = L.look_veto(part, start, 400, TRIM, 6000, 100, "bright", careful=False)
    assert careful.why == "shortened" and careful.max_ms < 3000
    assert careless == L.Veto(True, 6000, "unseen")  # the seen floor's end is no wall
    wall = L.VetoGrid()
    wall.mark_view(start, [(b, 70, 200, "wall") for b in (-260, 0, 260)], 0, "bright")
    blocked = L.look_veto(wall, start, 400, TRIM, 3000, 100, "bright", careful=False)
    assert blocked == L.Veto(False, 0, "seen_blocked")
    far = L.VetoGrid()
    far.mark_view(start, [(b, 70, 900, "thing") for b in (-260, 0, 260)], 0, "bright")
    before = L.look_veto(far, start, 400, TRIM, 6000, 100, "bright", careful=False)
    assert before.ok and L.MIN_SEGMENT_MS <= before.max_ms < 6000 and before.vx_cap_mm is None


def test_dark_frames_mark_nothing_and_a_new_epoch_forgets() -> None:
    grid = L.VetoGrid()
    grid.mark_view(L.Pose2(0, 0, 0), [(0, 70, 1500)], 1000, "dark")
    assert not grid.cells  # a dark frame proves nothing about the floor
    grid.mark_walked([L.Pose2(0, 0, 0), L.Pose2(300, 0, 0)], 1000, "dark")
    assert not grid.cells  # nor does walking in the dark
    seen = L.VetoGrid()
    _seen_corridor(seen, 1000, "bright")
    before = {k: (c.seen_ms, c.level) for k, c in seen.cells.items()}
    seen.mark_walked([L.Pose2(0, 0, 0), L.Pose2(300, 0, 0)], 500_000, "dark")
    assert {k: (c.seen_ms, c.level) for k, c in seen.cells.items()} == before  # no refresh
    assert not seen.fresh(300, 0, 1000 + 600_001, "dark")
    grid2 = L.VetoGrid()
    _seen_corridor(grid2, 1000, "bright")
    assert grid2.cells
    grid2.sync(1)
    assert not grid2.cells


def test_head_down_watch_fires_only_when_stuck() -> None:
    start = L.Pose2(0, 0, 0)
    moving = L.HeadDownWatch(400, 0)
    for t in range(0, 6000, 20):
        # 0.18 m/s along a curve turning 10°/s
        h = -10 * t // 57
        assert not moving.step(t, start, L.Pose2(180 * t // 1000, -(t * t) // 60_000, h))
    stuck = L.HeadDownWatch(400, 0)
    fired = [t for t in range(0, 6000, 20) if stuck.step(t, start, L.Pose2(10 + t // 400, 0, 0))]
    assert fired == [L.HeadDownConfig().grace_ms + L.HeadDownConfig().window_ms]
    slow_start = L.HeadDownWatch(400, 0)  # stalls after a good start
    fired = [
        t for t in range(0, 6000, 20) if slow_start.step(t, start, L.Pose2(min(t, 3000) // 5, 0, 0))
    ]
    assert fired and 3000 < fired[0] <= 4100


def test_look_run_waits_for_stillness_and_handles_a_retake() -> None:
    run = L.LookRun(L.CATALOGUE["glance_down"], "w1.pre", 0, yaw_mrad=200)
    head, asks, done = run.step(0, still=True, legs_quiet=True)
    assert head == L.HOME and not asks and not done
    head, asks, done = run.step(300, still=False, legs_quiet=True)
    assert head == (L.HALF_DOWN[0], L.HALF_DOWN[1], 200, 0) and not asks  # aimed, but not still
    head, asks, done = run.step(420, still=True, legs_quiet=True)
    assert len(asks) == 1 and isinstance(asks[0], LookRequest)
    assert asks[0].look_id == "w1.pre" and asks[0].kind == "glance_down"
    assert "floor" in asks[0].tasks
    _, again, _ = run.step(
        460,
        still=True,
        legs_quiet=True,
        results=[{"look_id": "w1.pre", "status": "retake", "retake_in_ms": 150}],
    )
    assert not again
    _, again, _ = run.step(500, still=True, legs_quiet=True)
    assert not again  # waits retake_in_ms
    _, again, _ = run.step(620, still=True, legs_quiet=True)
    assert len(again) == 1
    ok = {"look_id": "w1.pre", "status": "ok", "results": {"light": {"level": "bright"}}}
    head, _, done = run.step(700, still=True, legs_quiet=True, results=[ok])
    assert not done and run.results == [ok]
    head, _, done = run.step(700 + 10, still=True, legs_quiet=True)
    assert done and head == L.HOME


def test_look_run_scan_ids_and_gives_up_without_stillness() -> None:
    run = L.LookRun(L.CATALOGUE["scan"], "s9", 0)
    ids = []
    t = 0
    while not run.finished and t < 20_000:
        _, asks, _ = run.step(t, still=True, legs_quiet=True)
        for a in asks:
            ids.append(a.look_id)
            run.step(
                t + 20,
                still=True,
                legs_quiet=True,
                results=[{"look_id": a.look_id, "status": "ok"}],
            )
        t += 40
    assert ids == ["s9.0", "s9.1", "s9.2"]
    never = L.LookRun(L.CATALOGUE["end_look"], "e1", 0)
    t = 0
    while not never.finished and t < 10_000:
        never.step(t, still=False, legs_quiet=True)
        t += 40
    assert never.finished and never.missed == 1


def test_seated_looks_keep_the_seated_head() -> None:
    run = L.LookRun(L.CATALOGUE["glance_down"], "g", 0, seated=True)
    head, _, _ = run.step(400, still=False, legs_quiet=True)
    assert head[0] == 0 and head[1] == 0


def test_result_adapters() -> None:
    result = {
        "look_id": "w1.end",
        "status": "ok",
        "results": {
            "light": {"level": "dim"},
            "floor": {
                "free_mm": [900, 1200, 1500, 800, 400],
                "ends": ["wall", "unknown", "thing", "drop", "x"],
                "trip": {"mm": 350, "mrad": -40},
            },
            # duck-eye's nest task (eye.nesttag): the tag under ``anchor``, its ``yaw_mrad`` the
            # heading of the tag's face normal in the duck's level frame (π when square)
            "nest": {
                "ok": 1,
                "level": "dim",
                "anchor": {
                    "id": 17,
                    "r_mm": 1800,
                    "b_mrad": 120,
                    "e_mrad": -50,
                    "yaw_mrad": 3172,
                    "yaw_amb_mrad": 0,
                    "conf_ppm": 900_000,
                },
                "tags": [],
                "cue": None,
            },
        },
    }
    fold = L.floor_fold(result)
    assert fold == {
        "look_id": "w1.end",
        "light": "dim",
        "free_mm": [900, 1200, 1500, 800, 400],
        "ends": ["wall", "unknown", "thing", "drop", "unknown"],
        "trip": {"mm": 350, "mrad": -40},
    }
    sectors = L.view_sectors(result)
    assert len(sectors) == 5 and sectors[0][0] > sectors[-1][0]  # sector 0 is the left
    assert L.nest_fix(result) == {
        "seen": True,
        "src": "tag",
        "tag": 17,
        "b_mrad": 120,
        "r_mm": 1800,
        "e_mrad": -50,
        "yaw_mrad": -30,  # the duck is turned 30 mrad right of square-on
        "yaw_sure": True,
        "conf_ppm": 900_000,
    }
    nest = result["results"]["nest"]
    unsure = dict(nest, anchor=dict(nest["anchor"], yaw_amb_mrad=900))
    fix = L.nest_fix({**result, "results": {"nest": unsure}})
    assert fix is not None and fix["yaw_sure"] is False and fix["r_mm"] == 1800
    # a cue only (the status light, the cyan panel): a direction, no range (owner C5)
    cue = {"kind": "light", "b_mrad": -200, "e_mrad": 40, "px": 30, "conf_ppm": 600_000}
    only_cue = dict(nest, anchor=None, cue=cue)
    assert L.nest_fix({**result, "results": {"nest": only_cue}}) == {
        "seen": True,
        "src": "cue",
        "b_mrad": -200,
        "e_mrad": 40,
        "conf_ppm": 600_000,
    }
    assert L.floor_fold({"look_id": "x", "status": "none", "why": "timeout"}) is None
    assert L.light_of(None) == "none"
    none = {"ok": 1, "level": "dim", "anchor": None, "tags": [], "cue": None}
    assert L.nest_fix({"look_id": "x", "status": "ok", "results": {"nest": none}}) is None
    assert L.nest_fix({"look_id": "x", "status": "ok", "results": {}}) is None


# ── duck.look (LookExec) ──────────────────────────────────────────────────────


def _look_ctx(t: int, posture: str, eye: tuple[Any, ...] = (), still: bool = True) -> Any:
    from duckmind.duckbody.protocol import ActContext, PoseSample, ProprioView

    pose = PoseSample(
        seq=t // 20,
        t_ns=t * 1_000_000,
        trunk_height_mm=125,
        camera_pos=None,
        camera_quat=None,
        tof_pos=None,
        tof_quat=None,
        imu_quat=None,
        gyro=None,
        head=L.V5_HOME_HEAD,
        head_vel=0.0,
        odom=(0.5, 0.2, 0.3),
        epoch=0,
        still=still,
        policy=posture,
    )
    return ActContext(
        t_ns=t * 1_000_000,
        state=None,
        pose=pose,
        depth=None,
        proprio=ProprioView(posture=posture, policy=posture),
        light="bright",
        eye=eye,
    )


def _run_look(
    cmd: Any, posture: str = "stand", nest: bool = False
) -> tuple[dict[str, Any], list[Any], Any]:
    grid = L.VetoGrid()
    ex = L.LookExec(cmd, 0, grid=grid)
    pending: list[tuple[int, str]] = []
    asked = []
    heads = []
    t = 0
    while t < 20_000:
        ready = tuple(
            {
                "look_id": lid,
                "status": "ok",
                "results": {
                    "light": {"level": "bright"},
                    "floor": {"free_mm": [1200] * 5, "ends": ["wall"] * 5},
                    **(
                        {
                            "nest": {
                                "ok": 1,
                                "level": "bright",
                                "anchor": {
                                    "id": 17,
                                    "r_mm": 1500,
                                    "b_mrad": 80,
                                    "e_mrad": -70,
                                    "yaw_mrad": 3122,
                                    "yaw_amb_mrad": 0,
                                    "conf_ppm": 900_000,
                                },
                                "tags": [],
                                "cue": None,
                            }
                        }
                        if nest
                        else {}
                    ),
                },
            }
            for due, lid in pending
            if due <= t
        )
        pending = [(due, lid) for due, lid in pending if due > t]
        out = ex.step(_look_ctx(t, posture, ready))
        heads.append(out.head)
        for look in out.looks:
            asked.append(look)
            pending.append((t + 100, look.look_id))
        if out.done:
            return dict(out.percepts[0]), asked, grid
        t += 20
    raise AssertionError("look did not finish")


def test_look_exec_scan_is_a_valid_sense_scan() -> None:
    from duckmind import percept
    from duckmind.duckbody.protocol import LookCmd

    done, asked, grid = _run_look(LookCmd(act_id="k1", deadline_ms=10**9, kind="scan"))
    assert [a.look_id for a in asked] == ["k1.0", "k1.1", "k1.2"]
    assert all(a.kind == "scan" for a in asked)
    assert done["kind"] == "sense.scan" and done["head_mrad"] == [
        0,
        L.SCAN_LEFT_MRAD,
        L.SCAN_RIGHT_MRAD,
    ]
    assert len(done["sectors"]) == 15 and done["light"] == "bright"
    event = percept.make(
        "sense.scan",
        src="live",
        t_ms=done["t_ms"],
        pv="p:0123456789abcdef",
        **{k: v for k, v in done.items() if k not in ("kind", "t_ms")},
    )
    assert percept.validate(event) == []
    assert grid.cells  # the scan's free ground is remembered for look_veto


def test_look_exec_seek_nest_reports_the_sighting_and_sits_still() -> None:
    from duckmind import percept
    from duckmind.duckbody.protocol import LookCmd

    cmd = LookCmd(act_id="k2", deadline_ms=10**9, kind="seek_nest", want="nest", bearing_mrad=300)
    done, asked, _ = _run_look(cmd, posture="sit", nest=True)
    assert asked[0].kind == "seek_nest" and "nest" in asked[0].tasks
    assert asked[0].hints == {"target_b_mrad": 300, "nest_b_mrad": 300}
    sighting = done["sightings"][0]
    assert sighting["what"] == "nest" and sighting["r_mm"] == 1500 and sighting["r_src"] == "tag"
    assert sighting["yaw_mrad"] == 20 and sighting["id"] == "tag36h11:17"
    event = percept.make(
        "sense.scan",
        src="live",
        t_ms=done["t_ms"],
        pv="p:0123456789abcdef",
        **{k: v for k, v in done.items() if k not in ("kind", "t_ms")},
    )
    assert percept.validate(event) == []
    # seated: a glance_down becomes the seated look (rule 6)
    done2, asked2, _ = _run_look(
        LookCmd(act_id="k3", deadline_ms=10**9, kind="glance_down"), posture="sit"
    )
    assert asked2[0].kind == "fixate"


def test_look_exec_never_captures_while_moving() -> None:
    from duckmind.duckbody.protocol import LookCmd

    ex = L.LookExec(LookCmd(act_id="k4", deadline_ms=10**9, kind="fixate"), 0)
    for t in range(0, 2000, 20):
        out = ex.step(_look_ctx(t, "stand", still=False))
        assert not out.looks
