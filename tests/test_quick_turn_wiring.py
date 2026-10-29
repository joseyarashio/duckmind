"""2026-10-09 the quick turn as robotd skills: the guard's rules, the entry choice, the table, the body link's hold and hand-back."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_neuro_bodylink import DT, moves, state

from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import guard as G
from duckmind.skilltrain import motion_table as MT
from duckmind.skilltrain import table as T

NAMES = [*G.QUICK_TURN_ENTRIES, G.ABORT]


def test_the_entry_follows_the_side_of_the_turn() -> None:
    assert G.quick_turn_entry(1.2, NAMES) == "quick_turn_left"
    assert G.quick_turn_entry(-0.9, NAMES) == "quick_turn_right"
    assert G.quick_turn_entry(1.2, [G.ABORT]) is None  # a table without the quick-turn entries
    assert G.quick_turn_running(NAMES, "quick_turn_left") and G.quick_turn_running(
        NAMES, "quick_turn_right:unwind"
    )
    assert not G.quick_turn_running(NAMES, "walk")


def test_a_quick_turn_is_one_run_wear_free_and_ends_with_the_abort() -> None:
    g = G.SkillGuard(NAMES)
    view = lambda label, twist=0.4: G.BodyView(label, twist=twist, vx=twist, tilt_deg=2.0)  # noqa: E731
    assert g.request("quick_turn_left", view("walk"), 100.0).ok
    assert g.request("quick_turn_left", view("quick_turn_left"), 100.02).reason == "hold"
    assert (
        g.request("quick_turn_right", view("quick_turn_left"), 100.5).reason == "hold"
    )  # the other way: the same run
    assert g.counts(100.5)["per_skill"]["quick_turn_left"] == 0
    assert g.request(G.ABORT, view("quick_turn_right"), 101.0).ok
    assert (
        g.request("quick_turn_left", view("walk", twist=0.9), 110.0).reason == "speed"
    )  # gated from <= 0.5 m/s


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


def test_the_table_has_the_quick_turn_entries_under_the_gated_command_and_a_pin(
    installed: Path,
) -> None:
    doc = json.loads(installed.read_text())
    rows = {s["name"]: s for s in doc["skills"]}
    assert doc["files"]["qturn"]["sha256"] == MT.sha256_of(doc["files"]["qturn"]["path"])
    assert rows["quick_turn_left"]["command"] == [0.0, 0.0, MT.QTURN_YAW]
    assert rows["quick_turn_right"]["command"] == [0.0, 0.0, -MT.QTURN_YAW]
    for n in G.QUICK_TURN_ENTRIES:
        assert (
            rows[n]["chain"] is True
            and rows[n]["unwind_s"] == 0.0
            and rows[n]["duration"] == MT.QTURN_DURATION_S
        )
        assert rows[n]["action_scale"] == 0.9 and rows[n]["gain_ratio"] == 1.0
    assert len(T.resolve(T.load(installed))) <= 30
    Path(doc["files"]["qturn"]["path"]).write_bytes(b"another network")
    assert any("qturn" in p for p in MT.check(installed))


def _link() -> BodyLink:
    link = BodyLink()
    link.skills = G.SkillGuard(NAMES)
    return link


def _tick(link: BodyLink, t: float, label: str, req: BodyRequest, **kw):
    return link.tick(state(t, policy=label, **kw), req)


def _do(out) -> list[str]:
    return [i.params["skill"] for i in out.intents if i.method == "robot.do"]


def test_the_link_runs_a_pure_turn_as_the_quick_turn_holds_it_and_hands_back() -> None:
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest(vx=0.0, vyaw=0.0))
    turn = BodyRequest(
        vx=0.37, vyaw=0.9, pivot=1.0
    )  # the brain turns toward something far off its heading
    t += DT
    assert _do(_tick(link, t, "walk", turn)) == ["quick_turn_left"]
    for _ in range(20):  # robotd now says quick_turn_left: the link keeps asking
        t += DT
        out = _tick(link, t, "quick_turn_left", turn)
        assert _do(out) == ["quick_turn_left"] and "posture" not in out.gate
    t += DT
    assert _do(_tick(link, t, "quick_turn_left", BodyRequest(vx=0.37, vyaw=0.9))) == [
        G.ABORT
    ]  # the last stretch is the arc's
    t += DT
    assert _do(_tick(link, t, "walk", BodyRequest(vx=0.37, vyaw=0.9))) == []


def test_arcs_walks_and_flights_stay_v5s() -> None:
    link = _link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "walk", BodyRequest())
    for req in (
        BodyRequest(vx=0.0, vyaw=0.5),
        BodyRequest(vx=0.3, vyaw=1.0),
        BodyRequest(vx=0.5, vyaw=0.9, pivot=1.0, panic=True),
    ):
        t += DT
        assert "quick_turn_left" not in _do(_tick(link, t, "walk", req))


def test_without_the_quick_turn_entries_a_turn_stays_v5() -> None:
    link = BodyLink()
    link.skills = G.SkillGuard(["skill_abort", "startle_2"])
    out = None
    for k in range(60):
        out = _tick(link, k * DT, "walk", BodyRequest(vx=0.37, vyaw=0.9, pivot=1.0))
    assert out is not None and _do(out) == [] and moves(out.intents)


def test_senses_and_stub_call_a_quick_turn_walking() -> None:
    from duckmind.neuro import livestub
    from duckmind.world import neuro_senses

    s = state(1.0, policy="quick_turn_left")
    assert neuro_senses._posture(s, None) == "walk" and livestub._posture(None, s) == "walk"


# ── the roll the duck asked for is a lie-down on purpose (the flip trick) ────────────────────────


def _roll_link() -> BodyLink:
    link = BodyLink()
    link.skills = G.SkillGuard([G.ROLL_ENTRY, G.ABORT])
    return link


def _events(out) -> list[str]:
    return [str(e.get("kind")) for e in out.events]


def test_a_roll_the_duck_asked_for_is_no_fall_and_no_righting_until_it_is_upright_a_second() -> None:
    link = _roll_link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "stand", BodyRequest())
    seen: list[str] = []
    righting: list[str] = []
    # the roulade runs; the trunk goes over (robotd's fall flag rises), then velstand takes it back up
    for k in range(60):
        t += DT
        over = 10 <= k < 40
        out = _tick(link, t, "roulade", BodyRequest(), fallen=over, tilt=150.0 if over else 5.0)
        seen += _events(out)
        righting.append(out.righting)
    for k in range(40):  # robotd's walk label again, upright: the hold goes on for a second
        t += DT
        out = _tick(link, t, "stand", BodyRequest(), fallen=False, tilt=3.0)
        seen += _events(out)
        righting.append(out.righting)
    assert "proprio.fell" not in seen and "body.fell" not in seen
    assert set(righting) == {"idle"}  # no righting episode was opened
    assert "body.lay_down" in seen  # (the claim that was held back is said once)


def test_a_duck_still_down_after_the_roll_falls_as_any_duck_does() -> None:
    link = _roll_link()
    t = 0.0
    for _ in range(40):
        t += DT
        _tick(link, t, "stand", BodyRequest())
    for _ in range(20):
        t += DT
        _tick(link, t, "roulade", BodyRequest(), fallen=True, tilt=120.0)
    seen: list[str] = []
    righting: set[str] = set()
    for _ in range(80):  # the skill is over and the duck is still on its back: never "up" to the hold
        t += DT
        out = _tick(link, t, "stand", BodyRequest(), fallen=True, tilt=120.0)
        seen += _events(out)
        righting.add(out.righting)
    assert link.rolling_until_ns > 0  # (the hold waits for it to be upright; it is bounded)
    t += ROLL_LATE
    out = _tick(link, t, "stand", BodyRequest(), fallen=True, tilt=120.0)
    seen += _events(out)
    righting.add(out.righting)
    assert "proprio.fell" in seen or righting - {"idle"}


ROLL_LATE = 31.0
