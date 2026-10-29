"""Curiosity P1's parts: the primitives and their rules, the kick cone, outcomes and tables, the
perceive adapter and tracker, the calibration summary."""

from __future__ import annotations

import json
import math

import pytest

from duckmind.curiosity import cone, curio_calib, outcomes, prims
from duckmind.curiosity.calib import CALIB, TIERS, Patience, Window
from duckmind.curiosity.perceive import Fix, Odom, Seen, Tracker, fix_of, pick, sightings
from duckmind.eye.protocol import EyeResult

# ── primitives and the rules ───────────────────────────────────────────────────


def test_autonomous_skills_are_the_three() -> None:
    assert prims.AUTONOMOUS_SKILLS == {"ground_pick", "kick_left", "kick_right"}
    assert not prims.AUTONOMOUS_SKILLS & prims.FORBIDDEN_SKILLS


def test_every_primitive_is_in_the_trained_range() -> None:
    made = [
        prims.leg(900, 1500),
        prims.leg(-2000, 400),
        prims.creep(),
        prims.creep(500, 3000),
        prims.trim(-700),
        prims.bump(),
        prims.back(),
        prims.back_arc(True),
        prims.back_arc(False),
        prims.turn_arc(True),
    ]
    for prim in made:
        assert abs(prim.vx_mm) <= 400 and abs(prim.vyaw_mrad) <= 1000
        assert prim.vx_mm != 0
        assert 100 <= prim.ms <= 6000


@pytest.mark.parametrize(
    "twist, prim, problem",
    [
        ((500, 0), "leg", "|vx|"),
        ((400, 1200), "leg", "|wz|"),
        ((0, 500), "leg", "on the spot"),
        ((200, 0), "creep", "does not step"),
        ((-400, 0), "creep", "reversing"),
    ],
)
def test_forbidden_twists_are_caught(twist: tuple[int, int], prim: str, problem: str) -> None:
    found = prims.problems(
        twist=twist, do=None, head=None, prim=prim, ms_since_step=10_000, head_down_ok=False
    )
    assert any(problem in p for p in found), found


def test_reversing_only_in_back_and_back_arc() -> None:
    for prim in ("back", "back_arc"):
        assert (
            prims.problems(
                twist=(-400, 1000 if prim == "back_arc" else 0),
                do=None,
                head=None,
                prim=prim,
                ms_since_step=0,
                head_down_ok=False,
            )
            == []
        )


def test_skills_only_autonomous_and_only_standing_still() -> None:
    def check(do: str, twist: tuple[int, int] | None = None, since: int = 10_000) -> list[str]:
        return prims.problems(
            twist=twist, do=do, head=None, prim=None, ms_since_step=since, head_down_ok=False
        )

    assert check("kick_left") == []
    assert check("roulade") and check("crouch") and check("roller") and check("sit_toggle")
    assert any("walking" in p for p in check("kick_right", (400, 0)))
    assert any("after a step" in p for p in check("ground_pick", None, 100))


def test_head_down_walking_only_with_the_a2_exception() -> None:
    down = (-0.125, 0.175, 0.0, 0.0)
    assert prims.problems(
        twist=(400, 0), do=None, head=down, prim="creep", ms_since_step=0, head_down_ok=False
    )
    assert (
        prims.problems(
            twist=(400, 0), do=None, head=down, prim="creep", ms_since_step=0, head_down_ok=True
        )
        == []
    )


# ── the kick cone ────────────────────────────────────────────────────────────


def test_cone_clear_in_open_space_and_blocked_by_the_leash() -> None:
    ball = (0, 0)
    ok = cone.check(ball, 0, "left", (500, 900), leash=cone.Circle(0, 0, 3000))
    assert ok.ok and ok.worst_mm is not None and ok.worst_mm > 0
    blocked = cone.check(ball, 0, "left", (500, 900), leash=cone.Circle(0, 0, 800))
    assert not blocked.ok and blocked.why == "leash"


def test_cone_tries_the_other_foot() -> None:
    # a fragile thing ahead-left: the left foot's cone hits it, the right foot's does not
    keep = [cone.Circle(600, 600, 450, "fragile")]
    choice = cone.choose((0, 0), 0, "left", (400, 900), leash=None, keep_out=keep)
    assert choice.ok and choice.foot == "right"
    both = cone.choose(
        (0, 0), 0, "left", (400, 900), leash=None, keep_out=[cone.Circle(700, 0, 2000, "person")]
    )
    assert not both.ok


def test_cone_sector_is_sampled_across_its_width_and_depth() -> None:
    points = cone.sector_points((0, 0), 0, (500, 900))
    angles = {round(math.atan2(y, x), 2) for x, y in points}
    dists = {round(math.hypot(x, y)) for x, y in points}
    assert min(angles) < -0.3 and max(angles) > 0.3
    assert min(dists) == 500 and max(dists) == 900


# ── outcomes and tables ─────────────────────────────────────────────────────


def test_outcomes_from_perception() -> None:
    assert outcomes.classify_kick(None) == "missed"
    assert outcomes.classify_kick(20) == "missed"
    assert outcomes.classify_kick(120) == "moved"
    assert outcomes.classify_kick(700) == "rolled"
    assert outcomes.classify_peck(mouth_mrad=90, moved=0, reachable=True) == "in_beak"
    assert outcomes.classify_peck(mouth_mrad=-87, moved=60, reachable=True) == "squirted"
    assert outcomes.classify_peck(mouth_mrad=None, moved=0, reachable=False) == "unreachable"
    assert (
        outcomes.classify_bump(
            moved=0,
            h_before=120,
            h_after=40,
            progress_mm=60,
            expected_mm=60,
            witnessed=False,
            gap_after_mm=0,
        )
        == "tipped"
    )
    assert (
        outcomes.classify_bump(
            moved=5,
            h_before=120,
            h_after=120,
            progress_mm=5,
            expected_mm=60,
            witnessed=True,
            gap_after_mm=0,
        )
        == "stuck"
    )
    assert (
        outcomes.classify_bump(
            moved=5,
            h_before=120,
            h_after=120,
            progress_mm=5,
            expected_mm=60,
            witnessed=False,
            gap_after_mm=0,
        )
        == "nothing"
    )
    assert (
        outcomes.classify_bump(
            moved=0,
            h_before=None,
            h_after=None,
            progress_mm=60,
            expected_mm=60,
            witnessed=False,
            gap_after_mm=120,
        )
        == "missed"
    )
    assert outcomes.in_beak(-87 + 52) and not outcomes.in_beak(-87 + 40)
    assert not outcomes.in_beak(200, commanded_open=1.0)
    assert outcomes.gap_bin(170) == 4


def test_wilson_matches_known_values() -> None:
    lo, hi = outcomes.wilson(0, 30)
    assert lo == 0.0 and abs(hi - 0.1135) < 0.001  # 0/30 bounds a rate at 11.4 %
    lo, hi = outcomes.wilson(72, 72)
    assert abs(lo - 0.9493) < 0.001


def test_outcome_table_is_integer_json() -> None:
    table = outcomes.OutcomeTable(physics="live", perception="camera")
    for moved, outcome in ((700, "rolled"), (650, "rolled"), (0, "missed")):
        table.add(outcomes.Row("kick", "ball_slow", "normal", outcome, moved, 14000))
    data = table.to_json()
    assert data["schema"] == "curiosity-outcomes-v1" and data["n"] == 3
    cell = data["cells"][0]
    assert cell["outcomes"]["rolled"]["k"] == 2 and cell["outcomes"]["rolled"]["ppm"] == 666667
    text = json.dumps(data)
    assert "." not in text.replace('"', " ").split("note")[0].replace("curiosity-outcomes-v1", "")
    with pytest.raises(ValueError):
        table.add(outcomes.Row("kick", "ball_slow", "normal", "exploded"))


# ── perceive ────────────────────────────────────────────────────────────────


def _result(things: list[dict], status: str = "ok") -> dict:
    return EyeResult(
        look_id="C1.L1",
        attempt=0,
        kind="fixate",
        status=status,
        pv="p:0123456789abcdef",
        why=None if status == "ok" else "timeout",
        captured_at_ns=5_000_000_000,
        results={
            "light": {"level": "bright", "luma": 90, "noise_q": 2},
            "things": {"things": things},
        }
        if status == "ok"
        else {},
    ).to_json()


def test_sightings_read_positions_either_way() -> None:
    seen = sightings(
        _result(
            [
                {
                    "label": "ball",
                    "x_mm": 300,
                    "y_mm": 40,
                    "w_mm": 70,
                    "h_mm": 70,
                    "conf_ppm": 800_000,
                },
                {"colour": "orange", "r_mm": 500, "b_mrad": 0},
                {"x_mm": 1, "y_mm": 1},  # no identity: skipped
            ]
        )
    )
    assert [s.key for s in seen] == ["ball", "orange"]
    assert seen[0].t_ms == 5000 and seen[0].r_mm == 303
    assert seen[1].x_mm == 500 and seen[1].y_mm == 0
    assert sightings(_result([], status="none")) == []


def test_things_task_foot_ranges_become_centres() -> None:
    """duck-eye's things task gives the range to the foot (near edge): the centre is half the
    registered depth farther along the bearing."""
    from duckmind.curiosity.perceive import centre_of

    seen = sightings(
        _result(
            [
                {
                    "id": None,
                    "colour": "orange",
                    "b_mrad": 0,
                    "r_mm": 400,
                    "w_mm": 70,
                    "h_mm": 70,
                    "conf_ppm": 700_000,
                }
            ]
        )
    )
    assert seen[0].key == "orange" and seen[0].foot
    centre = centre_of(seen[0], 70)
    assert (centre.x_mm, centre.y_mm) == (435, 0) and not centre.foot
    assert centre_of(centre, 70) == centre  # a centre stays a centre


def test_odom_frames_round_trip() -> None:
    at = Odom(1000, -500, 785)
    for point in ((300, 0), (0, 300), (-120, 45)):
        world = at.to_world(*point)
        back = at.to_trunk(*world)
        assert abs(back[0] - point[0]) <= 1 and abs(back[1] - point[1]) <= 1
    later = Odom(1100, -400, 785)
    dx, dy, dyaw = later.delta(at)
    assert abs(dx - 141) <= 1 and abs(dy) <= 1 and dyaw == 0


def test_tracker_agreement_and_freshness() -> None:
    track = Tracker("ball")
    at = Odom(0, 0, 0)
    track.update(fix_of(Seen("L1", 1000, "ball", 400, 0), at))
    assert not track.agreed()
    track.update(fix_of(Seen("L2", 1100, "ball", 405, 0), at))
    assert not track.agreed()  # too soon (< 200 ms)
    track.update(fix_of(Seen("L3", 1400, "ball", 408, 3), at))
    assert track.agreed()
    assert track.fresh(1400 + 10_000) and not track.fresh(1400 + 10_001)
    track.update(Fix(900, 0, 1500, "L4", "ball"))  # it moved: a moving thing ages in 3 s
    assert track.moving and not track.fresh(1500 + 3001)


def test_pick_prefers_the_label_near_the_belief() -> None:
    seen = [
        Seen("L1", 0, "ball", 900, 0),
        Seen("L1", 0, "ball", 410, 10),
        Seen("L1", 0, "cup", 400, 0),
    ]
    assert pick(seen, "ball", (400, 0)).x_mm == 410
    assert pick(seen, "cup", None).key == "cup"
    assert pick(seen, "ball", (0, 2000)) is None


# ── calibration ─────────────────────────────────────────────────────────────


def test_windows_and_patience() -> None:
    w = Window((60, 100), (20, 60))
    assert w.centre == (80, 40)
    assert w.contains(80, 40) and w.contains(99, 59) and not w.contains(101, 40)
    assert w.contains(105, 40, tol_pct=130) and not w.contains(95, 40, tol_pct=10)
    assert Patience.from_wits(1500) == Patience("hasty", 130, 300, 8000, 1700)
    assert Patience.from_wits(8500) == Patience("careful", 80, 1000, 20000, 1000)
    assert Patience.from_wits(5000).tier == "normal"
    assert TIERS["normal"].frustrated().tol_pct == 90
    assert CALIB.step_mm["creep"] > 0 and CALIB.step_mm["leg1"] > CALIB.step_mm["creep"]


def test_calib_summary_of_records() -> None:
    records = [
        {
            "task": "kick",
            "foot": foot,
            "ahead": a,
            "side": s,
            "placed": {"ok": True},
            "moved_mm": 700.0 if 0.06 <= a <= 0.09 and s >= 0.03 else 0.0,
            "dir_rad": 0.4 if foot == "left" else -0.3,
            "max_tilt": 3.0,
            "fell": False,
        }
        for foot in ("left", "right")
        for a in (0.05, 0.065, 0.08, 0.095)
        for s in (0.01, 0.04, 0.07)
    ] + [
        {
            "task": "creep",
            "label": "creep",
            "vx": 0.4,
            "secs": 0.5,
            "head": "home",
            "ahead_m": 0.042,
            "left_m": 0.0,
            "dyaw_rad": -0.02,
            "max_tilt": 3.0,
            "fell": False,
        }
        for _ in range(5)
    ]
    summary = curio_calib.summarise(records, sources={"x.jsonl": "0" * 64})
    assert summary["kick_window"] == {"ahead_mm": [65, 80], "side_mm": [40, 70], "spots": 4}
    assert summary["kick"]["per_foot"]["left"]["dir_mrad_p10_p50_p90"] == [400, 400, 400]
    assert summary["steps"]["creep"]["ahead_mm_p10_p50_p90"] == [42, 42, 42]
    assert summary["steps"]["creep"]["falls"]["k"] == 0


# ── the 「用眼睛做」 helpers (owner A6) ──────────────────────────────────────────


def test_eye_demo_command_from_layout_things() -> None:
    from duckmind.curiosity.demo import describe_zh, eye_demo_command

    ball = {
        "id": "ball_felt",
        "zh": "毛線球",
        "cls": "ball_slow",
        "shape": "sphere",
        "radius_mm": 35,
        "verbs": ["look", "bump", "peck", "kick", "dribble"],
    }
    fin = {
        "id": "fin_fish",
        "cls": "handle_toy",
        "verbs": ["look", "bump", "peck", "grab", "carry"],
        "parts": [
            {"shape": "box", "size_mm": [80, 40, 20]},
            {"shape": "box", "size_mm": [60, 10, 50], "offset_mm": [0, 0, 20]},
        ],
    }
    kick = eye_demo_command(ball, "kick", act_id="C7", deadline_ms=99_000)
    assert kick.phase == "step" and kick.thing == "ball_felt"
    assert kick.params == {
        "verb": "kick",
        "label": "ball_felt",
        "cls": "ball_slow",
        "tier": "normal",
        "carry_ok": False,
        "h_mm": 70,
        "d_mm": 70,
    }
    grab = eye_demo_command(fin, "grab", act_id="C8", deadline_ms=99_000, label="green")
    assert grab.params["carry_ok"] is True and grab.params["d_mm"] == 10
    assert grab.params["h_mm"] == 70 and grab.params["label"] == "green"
    with pytest.raises(ValueError):
        eye_demo_command(ball, "grab", act_id="C9", deadline_ms=1)
    text = describe_zh(
        {"verb": "kick", "ended_by": "done", "outcome": "rolled", "moved_mm": 736, "ms": 25400},
        "毛線球",
    )
    assert text == "小鴨用自己的眼睛找到毛線球，踢一下：它滾走了（大約 74 公分）。花了 25 秒。"
    assert "太暗" in describe_zh({"verb": "kick", "ended_by": "dark"}, "毛線球")


def test_runner_summary_compares_the_ducks_report_with_truth() -> None:
    def rec(verb: str, outcome: str, truth: int, **step: object) -> dict:
        return {
            "verb": verb,
            "truth_moved_mm": truth,
            "refused_n": 0,
            "violations": [],
            "problems": ["perception must be camera or robot for Duck Core (…)"],
            "step": {
                "verb": verb,
                "outcome": outcome,
                "ended_by": "done",
                "ms": 20_000,
                "looks": 8,
                "fell": False,
                **step,
            },
        }

    records = [
        rec("kick", "rolled", 700, moved_mm=690),
        rec("kick", "rolled", 40, moved_mm=650),  # the report disagrees with truth
        rec("kick", "moved", 900, lost=True, moved_mm=None),  # gone, not found: cannot tell
        rec("peck", "squirted", 35),
    ]
    summary = curio_calib.runner(records)
    kick = summary["kick"]
    assert kick["worked_by_truth"]["k"] == 2 and kick["worked_by_truth"]["n"] == 3
    assert kick["report_agrees_with_truth"]["k"] == 1 and kick["report_agrees_with_truth"]["n"] == 2
    assert kick["report_cannot_tell"] == 1 and kick["report_problems"] == 0
    assert summary["peck"]["worked_by_truth"]["k"] == 1
    table = curio_calib.outcome_table([r["step"] for r in records], physics="t", perception="p")
    assert table["n"] == 4 and {c["verb"] for c in table["cells"]} == {"kick", "peck"}
