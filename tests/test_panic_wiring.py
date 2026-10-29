"""2026-10-09 the panic gait as robotd skills: the guard's rules, the entry choice, the table, the body link's hold and hand-back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_neuro_bodylink import DT, moves, state

from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import gt_wiring as W
from duckmind.skilltrain import guard as G
from duckmind.skilltrain import motion_table as MT
from duckmind.skilltrain import table as T

NAMES = [*G.PANIC_ENTRIES, G.ABORT]


def test_the_live_path_asks_for_the_straight_run_only() -> None:
    # live (Mac, 2026-10-09): flights that began as an arc tipped or felled the duck on the hand-back to v5
    assert G.ARCS_ON is False
    for yaw in (-1.0, -0.6, 0.0, 0.3, 0.6, 1.0):
        assert G.panic_entry(yaw, NAMES) == "panic_run"
        assert G.panic_entry(yaw, NAMES, "panic_run") == "panic_run"


def test_the_entry_follows_the_yaw_with_hysteresis(monkeypatch) -> None:
    monkeypatch.setattr(
        G, "ARCS_ON", True
    )  # the arc rule itself (kept for the day the arcs are live)
    assert G.panic_entry(0.0, NAMES) == "panic_run"
    assert G.panic_entry(0.3, NAMES) == "panic_run"  # not tight enough to leave the straight run
    assert G.panic_entry(0.6, NAMES) == "panic_arc_left"
    assert G.panic_entry(-0.6, NAMES) == "panic_arc_right"
    assert G.panic_entry(0.3, NAMES, "panic_arc_left") == "panic_arc_left"  # stays in the arc …
    assert (
        G.panic_entry(0.2, NAMES, "panic_arc_left") == "panic_run"
    )  # … until it is nearly straight
    assert G.panic_entry(-0.3, NAMES, "panic_arc_left") == "panic_run"
    assert G.panic_entry(0.0, ["skill_abort"]) is None  # a table without the panic entries
    assert G.panic_entry(0.9, ["panic_run"]) == "panic_run"  # no arc entry loaded: the straight run


def view(
    label: str, *, twist: float = 0.4, vx: float | None = None, tilt: float = 2.0
) -> G.BodyView:
    return G.BodyView(label, twist=twist, vx=vx if vx is not None else twist, tilt_deg=tilt)


def test_a_flight_starts_from_a_walk_is_one_run_and_ends_with_the_abort() -> None:
    g = G.SkillGuard(NAMES)
    assert g.request("panic_run", view("walk", twist=0.4), 100.0).ok
    assert (
        g.request("panic_run", view("panic_run"), 100.02).reason == "hold"
    )  # the refresh each tick
    # an arc takes the flight over: free (not counted, no gap), the same run
    assert g.request("panic_arc_left", view("panic_run"), 100.5).reason == "hold"
    assert g.counts(100.5)["per_skill"]["panic_run"] == 0  # a flight is not wear-budgeted
    assert g.request(G.ABORT, view("panic_arc_left"), 101.0).ok  # always allowed while it runs
    assert not g.request(G.ABORT, view("walk"), 101.1).ok  # nothing to abort


def test_a_flight_is_not_started_from_a_sprint_or_a_tilt() -> None:
    g = G.SkillGuard(NAMES)
    assert (
        g.request("panic_run", view("walk", twist=0.9), 10.0).reason == "speed"
    )  # gated from <= 0.55 m/s
    assert g.request("panic_run", view("walk", tilt=25.0), 10.1).reason == "tilt"
    assert g.request("panic_run", view("walk"), 20.0).ok
    assert g.request(
        "panic_run", view("walk"), 20.08
    ).ok  # no gap, no cap for a flight (it blinks and goes on)


def test_the_tilt_watch_hands_a_flight_back() -> None:
    g = G.SkillGuard(NAMES)
    assert g.watch(view("panic_run", tilt=10.0), 5.0) is None
    assert g.watch(view("panic_run", tilt=35.0), 5.02) == G.ABORT
    assert g.watch(view("panic_run", tilt=36.0), 5.04) is None  # once per run


@pytest.fixture()
def installed(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    sources = {}
    for role in MT.FILES:
        f = src / f"{role}.onnx"
        f.write_bytes(role.encode() * 8)
        sources[role] = f
    walk = tmp_path / "walkdir"
    walk.mkdir()
    for name in ("velstand.onnx", "alpha_sitstand.onnx"):
        (walk / name).write_bytes(name.encode())
    return MT.install(tmp_path / "out", sources, walk_dir=walk)


def test_the_table_has_the_panic_entries_under_the_gated_commands_and_a_pin(
    installed: Path,
) -> None:
    doc = json.loads(installed.read_text())
    rows = {s["name"]: s for s in doc["skills"]}
    assert doc["files"]["panic"]["sha256"] == MT.sha256_of(doc["files"]["panic"]["path"])
    assert rows["panic_run"]["command"] == [1.0, 0.0, 0.0] and rows["panic_run"]["chain"] is True
    assert (
        "panic_arc_left" not in rows and "panic_arc_right" not in rows
    )  # freed 2026-10-09 (the live path is straight)
    for n in ("panic_run",):
        assert rows[n]["action_scale"] == 0.9 and rows[n]["gain_ratio"] == 1.0
        assert rows[n]["duration"] == MT.PANIC_DURATION_S == 0.4 and rows[n]["unwind_s"] == 0.0
    assert (
        len(T.resolve(T.load(installed))) == 28
    )  # of robotd's 30: 25 + the two quick-turn entries + the roulade
    rules = W.guard_rules(doc)
    assert (
        rules["panic_run"].free_continue
        and rules["panic_run"].hold
        and rules["panic_run"].max_walk_mps == 0.60
    )
    Path(doc["files"]["panic"]["path"]).write_bytes(b"another network")
    assert any("panic" in p for p in MT.check(installed))  # a changed network refuses the table


def _link() -> BodyLink:
    link = BodyLink()
    link.skills = G.SkillGuard(NAMES)
    return link


def _tick(link: BodyLink, t: float, label: str, req: BodyRequest, **kw):
    return link.tick(state(t, policy=label, **kw), req)


def _do(out) -> list[str]:
    return [i.params["skill"] for i in out.intents if i.method == "robot.do"]


def test_the_link_runs_a_flight_as_the_panic_skill_holds_it_and_hands_back() -> None:
    link = _link()
    t = 0.0
    walk = BodyRequest(vx=0.5, panic=True)
    for _ in range(40):  # the duck walks (v5) a moment: no flight yet without the flag
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    asked = []
    out = _tick(link, t + DT, "walk", walk)
    t += DT
    asked += _do(out)
    assert asked == ["panic_run"]  # the tick the brain flees
    for _ in range(
        55
    ):  # robotd now says panic_run (a flight runs at least MIN_FLIGHT_S): the link keeps asking (a held skill) and walks on
        t += DT
        out = _tick(link, t, "panic_run", walk)
        assert _do(out) == ["panic_run"]
        assert not [g for g in out.gate if g in ("posture", "head")]
    # the flight ends (the brain stops fleeing): the link hands back to v5 with skill_abort at once
    t += DT
    out = _tick(link, t, "panic_run", BodyRequest(vx=0.4))
    assert _do(out) == [G.ABORT]
    t += DT
    out = _tick(link, t, "walk", BodyRequest(vx=0.4))
    assert _do(out) == [] and moves(out.intents)


def test_the_walking_head_envelope_does_not_stop_a_flight_but_still_stops_a_walk() -> None:
    wild = (-0.62, 0.0, -1.1, 0.0)  # what the panic gait does to the head at 1 m/s (measured)
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    out = _tick(link, t + DT, "walk", BodyRequest(vx=0.5, panic=True))
    t += DT
    assert _do(out) == ["panic_run"]
    for _ in range(10):
        t += DT
        out = _tick(link, t, "panic_run", BodyRequest(vx=0.5, panic=True), head=wild)
        assert "head" not in out.gate and _do(out) == ["panic_run"]
    link2 = _link()  # a plain walk with the same head: stopped by the envelope, as before
    for k in range(40):
        out = _tick(link2, k * DT, "walk", BodyRequest(vx=0.4), head=wild)
    assert "head" in out.gate


def test_a_stop_rule_ends_the_flight() -> None:
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.4))
    t += DT
    _tick(link, t, "walk", BodyRequest(vx=0.5, panic=True))
    for _ in range(10):
        t += DT
        _tick(link, t, "panic_run", BodyRequest(vx=0.5, panic=True))
    t += DT
    out = _tick(
        link, t, "panic_run", BodyRequest(vx=0.5, panic=True), fallen=True
    )  # tipped: righting owns it
    assert _do(out) == [G.ABORT]


def test_without_the_panic_entries_the_flight_stays_v5() -> None:
    link = BodyLink()
    link.skills = G.SkillGuard(["skill_abort", "startle_2"])
    out = None
    for k in range(60):
        out = _tick(link, k * DT, "walk", BodyRequest(vx=0.5, panic=True))
    assert (
        out is not None
        and _do(out) == []
        and moves(out.intents) == [(pytest.approx(0.5), pytest.approx(0.0))]
    )


def test_senses_and_stub_call_a_flying_duck_walking() -> None:
    from duckmind.neuro import livestub
    from duckmind.world import neuro_senses

    s = state(1.0, policy="panic_run")
    assert neuro_senses._posture(s, None) == "walk" and livestub._posture(None, s) == "walk"
    assert neuro_senses._posture(state(1.0, policy="skill_abort"), None) == "stand"
    assert (
        neuro_senses._posture(state(1.0, policy="startle_2"), None) == "unknown"
    )  # others: as before


# ── the live bug of 2026-10-09: the flag dropped when the brain's walk blinked, the flight was aborted at once ──


def _live_motor():
    from duckmind.neuro import motor as MO
    from duckmind.simweb.neuro_live import LiveMotor

    return LiveMotor(MO, None, 7)


def _drive(motor, cmds_frames, state_):
    out = []
    for t, (cmd, frame) in enumerate(cmds_frames):
        req, _ = motor(frame.t_ms, cmd, frame, state_)
        out.append(req)
    return out


def test_the_flight_flag_is_carried_through_the_motors_minimum_burst() -> None:
    """The brain starts a flight (vx 0.35, panic) and a tick later its walk blinks (vx 0, panic false: the look before it goes
    on, a reflex): the motor holds the burst for 0.8 s at the last twist, and the flight must be held with it. Before the
    fix the flag followed the brain's command alone: the body link aborted the panic skill 17 ms after it started and the
    3 s gap then locked the flight out (live :8791, 2026-10-09: 0.38 m in 12 s)."""
    from dataclasses import replace

    from duckmind.neuro.contract import BodySense, MotorCommand, SensorFrame

    motor = _live_motor()
    st = state(0.0, policy="walk")

    def frame(t, posture="stand"):
        return SensorFrame(t_ms=t, body=BodySense(posture=posture))

    def cmd(t, **kw):
        return MotorCommand(t_ms=t, behaviour="flee", bout=1, variant=1, **kw)

    reqs = []
    t = 0
    for _ in range(40):  # standing: the pause rule (0.45 s) is long over
        reqs.append(motor(t, cmd(t, posture="stand"), frame(t), st)[0])
        t += 20
    start = []
    for _ in range(3):  # the flight begins
        start.append(
            motor(t, cmd(t, vx=0.35, panic=True, posture="stand"), frame(t, "walk"), st)[0]
        )
        t += 20
    assert start[-1].vx > 0 and start[-1].panic
    held = []
    for _ in range(10):  # the brain's walk blinks: vx 0, no panic flag
        held.append(
            motor(t, replace(cmd(t, posture="stand"), vx=0.0, panic=False), frame(t, "walk"), st)[0]
        )
        t += 20
    assert all(r.vx > 0 for r in held), (
        "the motor holds the burst at the last twist (burst >= 0.8 s)"
    )
    assert all(r.panic for r in held), "... and the flight is held with it"
    later = []
    for _ in range(60):  # the burst is over (0.8 s): the walk stops, so does the flight
        later.append(
            motor(t, replace(cmd(t, posture="stand"), vx=0.0, panic=False), frame(t, "walk"), st)[0]
        )
        t += 20
    assert not later[-1].panic and later[-1].vx == 0.0


def test_a_flight_that_ends_and_begins_again_is_not_locked_out() -> None:
    g = G.SkillGuard(NAMES)
    assert g.request("panic_run", view("walk", twist=0.4), 100.0).ok
    assert g.request(G.ABORT, view("panic_run"), 100.02).ok
    # the brain's walk blinked and goes on 60 ms later: the guard says yes again (no gap, no hourly cap for a flight)
    assert g.request("panic_run", view("walk", twist=0.4), 100.08).ok
    for k in range(200):
        g.request(G.ABORT, view("panic_run"), 101.0 + k)
        assert g.request("panic_run", view("walk", twist=0.4), 101.5 + k).ok
