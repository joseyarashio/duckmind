"""A real duck whose body was never measured does not walk (docs/deploy/bring-up.md): without a valid calibration for
the board and the networks it runs (duckbody/calibration.py) the brain moves only the head and the beak — no walk, no
posture change, no trained skill — and says why in six languages; with one, it walks on that body's own levels. A
simulated body is never held."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_body_calibration import SEAT, WALK, bench
from test_neuro_bodylink import run
from test_neuro_live import drive, robot_state, stub_driver

from duckmind.duckbody import calibration as C
from duckmind.duckbody.proprio import ProprioConfig
from duckmind.i18n import CODES
from duckmind.i18n.catalog import read_catalog
from duckmind.neuro.bodylink import GATE_ZH, BodyLink, BodyRequest

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
LEG_METHODS = ("robot.do", "robot.pose")


def real_body(**info: Any) -> SimpleNamespace:
    """A real duck as the driver sees its body: the bridge's handshake words, no world."""
    said = {
        "backend": "real",
        "policies": {"set": "v8", "walk_sha256": WALK, "sitstand_sha256": SEAT},
        "robotd_config": {"board": "face-board-v1"},
        **info,
    }
    return SimpleNamespace(
        real=True,
        rehearsal=bool(said.get("rehearsal")),
        collision="real",
        bridge_info=lambda: dict(said),
        policy_hashes=lambda: {"walk_sha256": WALK, "sitstand_sha256": SEAT},
    )


def legs(ticks: list[list]) -> list[Any]:
    """Everything sent that moves the legs: a non-zero twist, a skill or a posture toggle, a body bow."""
    sent = [i for tick in ticks for i in tick]
    return [
        i
        for i in sent
        if (i.method == "robot.move" and i.twist != (0.0, 0.0)) or i.method in LEG_METHODS
    ]


def driver_on(body: Any):
    driver = stub_driver()
    driver._body = body
    driver.state = "running"
    driver._link = driver._new_link()
    return driver


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path))
    return tmp_path


# ── the body link: head only ──


def test_a_head_only_link_sends_nothing_to_the_legs_and_says_why() -> None:
    link = BodyLink()
    link.head_only = C.WHY_NONE
    nod = (0.0, 0.2, 0.3, 0.0)
    outs = run(link, BodyRequest(vx=0.45, vyaw=0.3, posture="stand", head=nod, mouth=0.5), 0.0, 4.0)
    assert legs([list(o.intents) for o in outs]) == []
    assert all("uncalibrated" in o.gate for o in outs)
    assert all(o.twist == (0.0, 0.0) for o in outs)
    methods = {i.method for o in outs for i in o.intents}
    assert {"robot.head", "robot.mouth"} <= methods  # the head and the beak are live
    # a seated duck asked to stand stays seated; a skill is not started; a fallen one is not righted by duckmind
    seated = run(link, BodyRequest(posture="stand", skill="kick_left"), 4.0, 5.0, policy="sit")
    assert legs([list(o.intents) for o in seated]) == []
    down = run(link, BodyRequest(), 9.0, 6.0, fallen=True, tilt=80.0)
    assert legs([list(o.intents) for o in down]) == []
    assert "uncalibrated" in GATE_ZH


def test_a_link_that_is_not_head_only_walks_as_before() -> None:
    link = BodyLink()
    outs = run(link, BodyRequest(vx=0.45, vyaw=0.3, posture="stand"), 0.0, 1.0)
    assert legs([list(o.intents) for o in outs])
    assert all("uncalibrated" not in o.gate for o in outs)


# ── the driver: a real body needs its own calibration ──


def test_a_real_duck_without_a_calibration_moves_only_its_head(home: Path) -> None:
    driver = driver_on(real_body())
    ticks = drive(driver, 40.0)
    sent = [i for tick in ticks for i in tick]
    assert legs(ticks) == []
    assert "robot.head" in {i.method for i in sent}
    lock = driver.real_view()["walk_lock"]
    assert lock["why"] == C.WHY_NONE and lock["reason"].key == "livebody.real_uncalibrated"
    assert lock["reason"].params == {"why": C.WHY_NONE}


def test_a_simulated_body_is_never_held(home: Path) -> None:
    driver = driver_on(SimpleNamespace(real=False, collision="allcollisions"))
    assert legs(drive(driver, 40.0))
    assert driver._link.head_only is None and driver.real_view() is None
    assert driver._link.proprio.cfg.lift_load == ProprioConfig().lift_load


def test_with_its_calibration_a_real_duck_walks_on_its_own_levels(home: Path) -> None:
    C.save(C.parse(bench()))
    driver = driver_on(real_body())
    link = driver._link
    assert link.head_only is None and driver.real_view()["walk_lock"] is None
    assert link.proprio.cfg.lift_load == 240.0 and link.proprio.cfg.floor_band == (335.0, 1440.0)
    tracker_config = driver._depth_config
    assert (tracker_config.touch_zones, tracker_config.too_close_mm) == (40, 150)
    # (a duck whose legs draw what the document was measured at: standing on a floor, by its own levels)
    ticks = [driver.tick(robot_state(k * 0.02, load=420.0)) for k in range(2000)]
    assert legs(ticks)


@pytest.mark.parametrize(
    ("setup", "why"),
    [
        ("other", C.WHY_OTHER),  # the duck's networks were updated after the bench session
        ("broken", C.WHY_INVALID),
        ("no_board", C.WHY_UNKNOWN_BODY),
    ],
)
def test_a_calibration_for_other_networks_or_a_broken_one_does_not_count(
    home: Path, setup: str, why: str
) -> None:
    body = real_body()
    if setup == "other":
        C.save(C.parse(bench(sitstand_sha256="d" * 64)))
    elif setup == "broken":
        path = C.folder() / C.file_name("face-board-v1", WALK, SEAT)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(bench(proprio={"lift_load": 1.0})), encoding="utf-8")
    else:
        C.save(C.parse(bench()))
        body = real_body(robotd_config=None)  # (the bridge's user cannot read robotd.toml)
    driver = driver_on(body)
    assert driver._link.head_only == why
    assert legs(drive(driver, 20.0)) == []
    assert driver.real_view()["walk_lock"]["why"] == why


def test_a_rehearsal_is_held_like_a_real_duck_until_it_is_measured(home: Path) -> None:
    body = real_body(rehearsal=True, robotd_config=None)
    assert driver_on(body)._link.head_only == C.WHY_NONE
    C.save(C.parse(bench(board=C.REHEARSAL_BOARD)))
    assert driver_on(body)._link.head_only is None


# ── the words, in six languages ──


def test_the_reason_is_said_in_six_languages() -> None:
    for code in CODES:
        core = read_catalog(code, "core") or {}
        neuro = read_catalog(code, "neuro") or {}
        page = read_catalog(code, "page") or {}
        assert "livebody.real_uncalibrated" in core, code
        assert "neuro.gate.uncalibrated" in neuro, code
        assert "page.body.state.real_nowalk" in page, code
        for why in C.WHYS:  # every reason has its own words
            assert f"{why} {{" in core["livebody.real_uncalibrated"], (code, why)
            assert f"{why} {{" in page["page.body.state.real_nowalk"], (code, why)


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="needs Node")
def test_the_real_duck_card_says_it_is_not_walking_yet_and_why() -> None:
    link = {"url": "remote://duck.local:8797", "up": True, "rtt_ms_p50": 30.0, "deadman_ms": 1000}
    real = {"rehearsal": False, "refuses": [], "turn_on": None}
    base = {"enabled": True, "available": True, "remote": link}
    answers = {
        "locked": {**base, "real": {**real, "walk_lock": {"why": "none"}}},
        "updated": {**base, "real": {**real, "walk_lock": {"why": "other_networks"}}},
        "free": {**base, "real": {**real, "walk_lock": None}},
        # (not on yet: turning it on comes first)
        "off": {**base, "real": {**real, "turn_on": "…", "walk_lock": {"why": "none"}}},
    }
    script = (
        f"const api = require({json.dumps(str(STATIC / 'bodylink.js'))});"
        f"const answers = {json.dumps(answers)};"
        "const out = {};"
        "for (const [name, answer] of Object.entries(answers)) out[name] = api.linkState(answer);"
        "console.log(JSON.stringify(out));"
    )
    ran = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stderr
    said = json.loads(ran.stdout)
    assert said["locked"]["key"] == "page.body.state.real_nowalk"
    assert said["locked"]["params"] == {"url": link["url"], "why": "none"}
    assert said["updated"]["params"]["why"] == "other_networks"
    assert said["free"]["key"] == "page.body.state.real"
    assert said["off"]["key"] == "page.body.state.real_off"
