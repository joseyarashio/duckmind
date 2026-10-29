"""2026-10-05 trained-motion wiring: the one table, the page's registry, the guard's new rules, the scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from duckmind.neuro import motions as MO
from duckmind.skilltrain import gt_wiring as W
from duckmind.skilltrain import guard as G
from duckmind.skilltrain import motion_table as MT
from duckmind.skilltrain import table as T

ROLES = tuple(MT.FILES)


@pytest.fixture()
def installed(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    sources = {}
    for role in ROLES:
        f = src / f"{role}.onnx"
        f.write_bytes(role.encode() * 8)
        sources[role] = f
    walk = tmp_path / "walkdir"
    walk.mkdir()
    for name in ("velstand.onnx", "alpha_sitstand.onnx"):
        (walk / name).write_bytes(name.encode())
    return MT.install(tmp_path / "out", sources, walk_dir=walk)


def test_the_table_fits_robotds_30_with_the_official_roulade_in(installed: Path) -> None:
    doc = json.loads(installed.read_text())
    entries = T.load(installed)
    resolved = T.resolve(entries)
    # kick_left, kick_right, the roulade + 25 of ours (march, the mat steps and the panic arcs freed 2026-10-09; + the two quick turns)
    assert len(resolved) == 28 <= T.MAX_SKILLS
    assert doc["policy"]["none_slots"] == []  # (no slot of robotd's is switched off)
    assert MT.check(installed) == []
    assert "roulade" in T.skill_names_from_env({T.SKILLS_ENV: str(installed)})
    roll = next(d for d in resolved if d.name == "roulade")
    assert roll.path == doc["files"]["roulade"]["path"]  # the entry runs the pinned file …
    assert (roll.duration, roll.chain, roll.unwind_s) == (MT.ROULADE_DURATION_S, True, 0.0)
    assert (roll.action_scale, roll.gain_ratio) == (1.0, 1.0)  # … at the values the roll was trained with
    Path(doc["files"]["roulade"]["path"]).write_bytes(b"another roll")
    assert any("roulade" in p for p in MT.check(installed))  # a changed network refuses the table


def test_the_roll_rule_reaches_the_guard_from_the_table(installed: Path) -> None:
    rules = W.guard_rules(json.loads(installed.read_text()))
    roll = rules["roulade"]
    assert (roll.per_hour, roll.min_gap_s, roll.settle_s, roll.rise_settle_s) == (4, 120.0, 1.5, 1.5)
    assert roll.postures == frozenset({"stand"}) and roll.abort_tilt_deg == 360.0


def test_every_motion_has_its_entries_in_the_table(installed: Path) -> None:
    have = {e.name for e in T.load(installed) if not e.skill.removed}
    for m in MO.MOTIONS:
        assert set(m.needs()) <= have, m.id
        assert m.id in {row["id"] for row in MO.status(have)}
    assert all(row["installed"] for row in MO.status(have))
    assert not all(row["installed"] for row in MO.status({"startle_2"}))


def test_a_changed_network_refuses_the_table(installed: Path) -> None:
    doc = json.loads(installed.read_text())
    Path(doc["files"]["bow"]["path"]).write_bytes(b"other")
    assert any("bow" in p for p in MT.check(installed))


def test_the_default_install_sets_the_delay_and_only_a_checked_table(
    installed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MT, "DEFAULT_DIR", installed.parent)
    env: dict[str, str] = {}
    info = MT.install_defaults(env)
    assert env[MT.DELAY_ENV] == "20" and env[T.SKILLS_ENV] == str(installed) and info["why"] is None
    # an explicit delay and an explicit "none" win
    env = {MT.DELAY_ENV: "0", T.SKILLS_ENV: "none"}
    MT.install_defaults(env)
    assert env[MT.DELAY_ENV] == "0" and T.SKILLS_ENV not in env
    # a changed network: no table, with the reason
    doc = json.loads(installed.read_text())
    Path(doc["files"]["hop"]["path"]).write_bytes(b"x")
    env = {}
    info = MT.install_defaults(env)
    assert T.SKILLS_ENV not in env and "hop" in info["why"]


def rules_for(installed: Path) -> G.SkillGuard:
    return W.skill_guard(
        [e.name for e in T.load(installed) if not e.skill.removed], {T.SKILLS_ENV: str(installed)}
    )


def view(label: str = "stand", **kw) -> G.BodyView:
    return G.BodyView(label, **kw)


def test_the_b_skills_obey_their_gate_postures_and_speeds(installed: Path) -> None:
    guard = rules_for(installed)
    for t in range(40):
        guard.observe(view(), t * 0.1)
    assert guard.request("bow", view(), 5.0).ok
    sat = guard.request("lie_low", view("sit"), 6.0)
    assert not sat.ok and sat.reason == "posture"
    assert str(sat.reason_zh) == "現在坐著，這個動作只在站著時才做（只有這些姿勢測過）"
    fast = guard.request("turn_left", view("walk", twist=0.5, vx=0.5), 7.0)
    assert not fast.ok and fast.reason == "speed"
    assert guard.request("bow_release", view("stand"), 7.0).reason == "nothing"


def test_a_press_skips_the_wear_caps_but_not_the_safety_rules(installed: Path) -> None:
    guard = rules_for(installed)
    for t in range(40):
        guard.observe(view(), t * 0.1)
    assert guard.request("hop_back", view(), 5.0).ok
    again = guard.request("hop_back", view(), 6.0)
    assert not again.ok and again.reason == "gap"
    assert guard.request("hop_back", view(), 6.0, manual=True).ok
    assert guard.request("hop_back", view(tilt_deg=30.0), 7.0, manual=True).reason == "tilt"
    assert guard.request("hop_back", view(held=True), 7.0, manual=True).reason == "held"


def test_lying_low_is_not_aborted_at_the_bows_own_tilt(installed: Path) -> None:
    guard = rules_for(installed)
    assert guard.watch(view("lie_low", tilt_deg=32.0), 1.0) is None
    assert guard.watch(view("startle_2", tilt_deg=32.0), 2.0) == G.ABORT
    assert guard.watch(view("lie_low", tilt_deg=52.0), 3.0) == G.ABORT


def run_script(picker: G.SkillPicker, labels: list[str], start: float = 0.0) -> list[str | None]:
    out = []
    for i, label in enumerate(labels):
        out.append(
            picker.pick(
                reflex=None,
                behaviour="wander",
                bout=1,
                intensity=0.0,
                posture="stand",
                now_s=start + i * 0.02,
                label=label,
            )  # fmt: skip
        )
    return out


def test_a_held_page_run_is_asked_every_tick_then_ended_by_its_release() -> None:
    picker = G.SkillPicker(available=frozenset({"lie_low", G.BOW_RELEASE, G.ABORT}))
    picker.play("lie_low", 0.0, hold_s=0.5, end=G.BOW_RELEASE)
    labels = ["stand"] + ["lie_low"] * 40
    got = run_script(picker, labels)
    assert got[0] == "lie_low" and set(got[1:25]) == {"lie_low"}
    assert G.BOW_RELEASE in got and got.index(G.BOW_RELEASE) >= 25
    assert picker.script is None or picker.script.ending


def test_a_one_shot_page_run_is_asked_once_and_given_up_when_refused() -> None:
    picker = G.SkillPicker(available=frozenset({"shake_1"}))
    picker.play("shake_1", 0.0)
    got = run_script(picker, ["stand"] * 40)
    assert got[0] == "shake_1" and got.count("shake_1") == 1 and picker.script is None


def test_a_greeting_bow_and_the_shake_after_a_fall_come_from_the_brain() -> None:
    names = frozenset({"bow", "shake_1"})
    picker = G.SkillPicker(seed=3, available=names)
    pick = picker.pick(reflex=None, behaviour="greet", bout=2, intensity=0.5, posture="stand",
                       now_s=1.0, label="stand")  # fmt: skip
    assert pick == "bow"
    # after righting: a shake_1 asked again while the guard says wait (patience), when it chances
    shakes = 0
    for seed in range(20):
        p = G.SkillPicker(seed=seed, available=names)
        p.pick(reflex=None, behaviour="righting", bout=1, intensity=0.0, posture="other",
               now_s=0.0, label="stand")  # fmt: skip
        first = p.pick(reflex=None, behaviour="wander", bout=2, intensity=0.0, posture="rise",
                       now_s=1.0, label="rise")  # fmt: skip
        if first == "shake_1":
            shakes += 1
            assert p.pick(reflex=None, behaviour="wander", bout=2, intensity=0.0, posture="stand",
                          now_s=3.0, label="stand") == "shake_1"  # still asking: patience  # fmt: skip
    assert 5 <= shakes <= 19


def test_a_corner_turns_on_the_spot_to_the_open_side() -> None:
    picker = G.SkillPicker(available=frozenset({"turn_left", "turn_right", G.ABORT}))
    first = picker.pick(reflex=None, behaviour="wander", bout=1, intensity=0.0, posture="stand",
                        now_s=1.0, label="stand", corner_turn=-1)  # fmt: skip
    assert first == "turn_right"
    assert picker.script is not None and picker.script.end == G.ABORT


class Tick:
    def __init__(self, t_ms: int, policy: str, z: float = 0.115, heading: float = 0.0,
                 x: float = 0.0, tilt: float = 2.0, fallen: bool = False) -> None:  # fmt: skip
        self.t_ms, self.policy, self.fallen, self.tilt_deg = t_ms, policy, fallen, tilt
        self.trunk, self.trunk_t = (x, 0.0, z, heading), float(t_ms)
        self.head = (0.0, 0.0, 0.0, 0.0)


def test_the_page_says_what_a_turn_and_a_fall_did() -> None:
    from duckmind.skilltrain import motion_result as MR

    ticks = [Tick(i * 20, "stand") for i in range(10)]
    ticks += [Tick(200 + i * 20, "turn_left", heading=i * 0.05) for i in range(100)]
    ticks += [Tick(2200 + i * 20, "stand", heading=5.0) for i in range(40)]
    done = MR.summarise(ticks, 200, "turn_left", "turn")
    assert (
        done["numbers"]["ran"] and done["numbers"]["ended_standing"] and not done["numbers"]["fell"]
    )
    assert 250 <= abs(done["numbers"]["turned_deg"]) <= 300
    assert "原地轉了" in str(done["zh"])
    fell = ticks[:60] + [Tick(1400, "turn_left", fallen=True, tilt=80.0)]
    got = MR.summarise(fell, 200, "turn_left", "turn")
    assert got["numbers"]["fell"] and not got["numbers"]["ended_standing"]
    assert "跌倒" in str(got["tail_zh"])
    none = MR.summarise([Tick(i * 20, "stand") for i in range(30)], 100, "bow", "bow")
    assert not none["numbers"]["ran"] and "沒有動起來" in str(none["zh"])
