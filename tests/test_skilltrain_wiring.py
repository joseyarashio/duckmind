"""The Stage-1 skill wiring (tr:harness, 2026-09-28) in the neural duck's body path.

The launcher appends the ``[[policy.skill]]`` table of ``$DUCKMIND_ROBOTD_SKILLS`` to this
simulator's robotd.toml; the body link starts a trained skill with ``robot.do`` only through the
skill guard (postures, upright, no hand, wear caps), duck-body's intent guard lets exactly those
names through, and a running skill that tips past 30° is handed back to velstand.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

from duckmind.duckbody import protocol as P
from duckmind.duckbody.reflex import GuardContext, check_intent
from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.sim_live import POLICY_FILES, _policy_config
from duckmind.skilltrain import guard as SG
from duckmind.skilltrain import table as T

HOME_JOINTS = (0.3491, 0.3491, 0.0, 0.0)


def state(t: float, *, policy: str = "walk", tilt: float = 2.0, fallen: bool = False) -> RobotState:
    joints = [0.0] * 15
    for i, home in enumerate(HOME_JOINTS):
        joints[5 + i] = home
    r = math.radians(tilt)
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {
                "fallen": fallen,
                "limp": False,
                "gravity": [math.sin(r), 0.0, -math.cos(r)],
            },
            "joints": joints,
            "currents_ma": [4.5] * 15,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
        }
    )


def dos(out) -> list[str]:
    return [i.params.get("skill") for i in out.intents if i.method == "robot.do"]


def settled_link() -> BodyLink:
    link = BodyLink()
    link.skills = SG.SkillGuard(SG.DEFAULT_RULES)
    for k in range(30):  # a few quiet ticks standing
        link.tick(state(k * 0.02), BodyRequest())
    return link


def test_a_skill_request_starts_through_the_guard_and_the_intent_passes_duck_bodys_check() -> None:
    link = settled_link()
    out = link.tick(state(0.62), BodyRequest(skill="startle_2"))
    assert dos(out) == ["startle_2"]
    assert any(e.get("kind") == "skill.started" for e in out.events)
    ctx = GuardContext(trained_skills=link.skills.names)
    assert check_intent(P.Intent.do("startle_2"), ctx) is None
    # without the guard's say-so duck-body still refuses anything outside its old list
    assert check_intent(P.Intent.do("startle_2"), GuardContext()) is not None


def test_a_seated_duck_is_refused_and_the_page_is_told_why() -> None:
    link = settled_link()
    out = link.tick(state(0.62, policy="sit"), BodyRequest(skill="startle_3"))
    assert dos(out) == [] and "skill" in out.gate
    refused = [e for e in out.events if e.get("kind") == "skill.refused"]
    assert refused and refused[0]["why"] == "posture"


def test_a_running_skill_that_tips_goes_back_to_velstand() -> None:
    link = settled_link()
    link.tick(state(0.62), BodyRequest(skill="freeze_4"))
    out = link.tick(state(0.64, policy="freeze_4", tilt=35.0), BodyRequest())
    assert dos(out) == ["skill_abort"]
    again = link.tick(state(0.66, policy="freeze_4:unwind", tilt=40.0), BodyRequest())
    assert dos(again) == []  # once per run


def test_no_guard_no_skills() -> None:
    link = BodyLink()
    out = link.tick(state(0.0), BodyRequest(skill="startle_2"))
    assert dos(out) == []


def test_the_launcher_appends_a_parseable_skill_table(tmp_path: Path, monkeypatch) -> None:
    pol = tmp_path / "pol"
    pol.mkdir()
    for name in POLICY_FILES:
        (pol / name).write_bytes(b"x")
    nets = {}
    for name in ("startle", "comfort"):
        nets[name] = tmp_path / f"{name}.onnx"
        nets[name].write_bytes(b"x")
    table = T.save(
        tmp_path / "skills.json",
        T.stage1(str(nets["startle"]), str(nets["comfort"]), str(pol / "velstand.onnx")),
    )
    monkeypatch.setenv(T.SKILLS_ENV, str(table))
    text = (
        _policy_config(pol) + "\n[control]\nhead_alpha = 1.0\n" + T.robotd_skill_section_from_env()
    )
    doc = tomllib.loads(text)
    names = [s["name"] for s in doc["policy"]["skill"]]
    assert names[:4] == ["startle_1", "startle_2", "startle_3", "startle_4"]
    assert doc["policy"]["walk"].endswith("velstand.onnx") and doc["control"]["head_alpha"] == 1.0
    assert set(T.skill_names_from_env()) == set(names)


# ── s1b:fix: held crouches, speed limits, quiet refusals ─────────────────────────────────────


def moving_state(t: float, *, policy: str = "walk", vx: float = 0.0) -> RobotState:
    s = state(t, policy=policy)
    return RobotState(**{**s.__dict__, "applied": (vx, 0.0, 0.0)})


def test_a_held_crouch_is_re_requested_quietly_and_ends_with_the_release() -> None:
    link = settled_link()
    out = link.tick(state(0.62), BodyRequest(skill="startle_2"))
    assert dos(out) == ["startle_2"]
    # the brain's freeze follows: the hold takes over the running startle (no rise between)
    out = link.tick(state(0.64, policy="startle_2"), BodyRequest(skill="freeze_2"))
    assert dos(out) == ["freeze_2"]
    started = [e for e in out.events if e.get("kind") == "skill.started"]
    assert started and started[0]["how"] == "continue"
    events = []
    for k in range(100):  # 2 s of re-requests: every tick a robot.do, no event
        out = link.tick(state(0.66 + 0.02 * k, policy="freeze_2"), BodyRequest(skill="freeze_2"))
        assert dos(out) == ["freeze_2"]
        events += [e for e in out.events if str(e.get("kind", "")).startswith("skill.")]
    assert events == []
    out = link.tick(state(2.70, policy="freeze_2"), BodyRequest(skill=SG.RELEASE))
    assert dos(out) == [SG.RELEASE]
    assert link.skills is not None and link.skills.counts(3.0)["per_skill"]["freeze_2"] == 1


def test_the_speed_limit_and_a_refusal_said_once() -> None:
    link = BodyLink()
    rules = dict(SG.DEFAULT_RULES)
    rules["freeze_4"] = SG.Rule(**{**rules["freeze_4"].__dict__, "max_walk_mps": 0.40})
    link.skills = SG.SkillGuard(SG.DEFAULT_RULES, rules)
    for k in range(30):
        link.tick(moving_state(k * 0.02, vx=0.55), BodyRequest())
    said = []
    for k in range(10):
        out = link.tick(moving_state(0.62 + 0.02 * k, vx=0.55), BodyRequest(skill="freeze_4"))
        assert dos(out) == []
        said += [e for e in out.events if e.get("kind") == "skill.refused"]
    assert len(said) == 1 and said[0]["why"] == "speed"
