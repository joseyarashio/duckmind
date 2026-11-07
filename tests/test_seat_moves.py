"""robotd 0.16.0+ starts no move from the seat ("one move at a time, and none from the seat", pollen-robotics/microduck
f93ecf7: a network started from a pose it was not trained from), and says so only in its own log. On v8 (robotd 0.18.5)
the brain kept asking for its seated stretch (``stretch_sit``): in the V8Soak it never played (v5: 15 times in 4 h 15 min).
On such a robotd the brain asks no skill from the seat; a seated stretch stands up first and stretches standing
(``stretch_tall``, trained from standing), then sits back down. v5 (0.15.0) keeps its seated stretch
(UPSTREAM-0185-2026-11-05 §13)."""

from __future__ import annotations

from test_neuro_bodylink import DT, state
from test_neuro_live import stub_driver

from duckmind.duckbody.robotd_client import RobotdClient
from duckmind.neuro import contract as C
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import guard as G

NAMES = ["sit_toggle", "stretch_sit", "stretch_tall", "shake_1", "startle_2", G.ABORT]


def client(version: str | None) -> RobotdClient:
    out = RobotdClient(None, None, None)  # type: ignore[arg-type]
    out.hello = {"api_version": 49, "daemon_version": version}
    out.capabilities = {"skills": list(NAMES)}
    return out


def test_the_daemon_says_whether_a_move_starts_from_the_seat() -> None:
    c = client(None)
    for version, seated in (("0.15.0", True), ("0.15.3", True), ("0.16.0", False), ("0.18.5", False),
                            (None, False)):  # fmt: skip
        c.hello = {"api_version": 49, "daemon_version": version}
        assert c.moves_from_seat is seated, version


def test_the_guard_refuses_a_seated_entry_on_a_robotd_that_starts_nothing_from_the_seat() -> None:
    seated = G.BodyView("sit", twist=0.0, gyro_rad_s=0.0)
    for guard, ok in (
        (G.SkillGuard(NAMES), True),
        (G.SkillGuard(NAMES, seated_moves=False), False),
    ):
        for k in range(200):  # seated 4 s (the seated stretch's 3 s settle)
            guard.watch(seated, k * DT)
        d = guard.request("stretch_sit", seated, 4.0)
        assert d.ok is ok and (ok or d.reason == "seat")


def _live(version: str, behaviour: str = "stretch_yawn", seconds: float = 16.0):
    """A seated duck whose brain starts ``behaviour`` 4 s in (and wants to stay seated), through the driver's skill
    request and the real body link; a robotd stand-in plays the sit-stand toggle (a 1 s rise), the skills (2 s) and
    refuses — as 0.16+ does — any skill asked while it sits. Returns (skills asked while seated, skills started,
    the labels seen)."""
    driver = stub_driver()
    driver._client = client(version)
    link = BodyLink()
    label, until = "sit", 0.0
    asked_seated: list[str] = []
    started: list[str] = []
    labels: list[str] = []
    t = 0.0
    while t < seconds:
        t += DT
        if until and t >= until:
            label, until = "walk", 0.0  # the rise or the skill is over: velstand has it
        st = state(t, policy=label, load=30.0 if label == "sit" else 45.0)
        bout = 1 if t >= 4.0 else 0
        cmd = C.MotorCommand(
            t_ms=int(t * 1000),
            behaviour=behaviour if bout else "rest_sit",
            bout=bout,
            variant=0,
            posture="sit",
        )
        request = driver._skill_request(
            link, cmd, st, BodyRequest(posture="sit"), None, int(t * 1000)
        )
        out = link.tick(st, request)
        for intent in out.intents:
            if intent.method != "robot.do":
                continue
            skill = intent.params["skill"]
            if skill == "sit_toggle":
                if label == "sit":
                    label, until = "rise", t + 1.0
                elif label == "walk":
                    label = "sit"
            elif label == "sit":
                asked_seated.append(skill)
                if version.startswith("0.15"):
                    started.append(skill)
                    label, until = skill, t + 2.0
            elif label == "walk":
                started.append(skill)
                label, until = skill, t + 2.0
        labels.append(label)
    return asked_seated, started, labels


def test_v8_asks_no_move_from_the_seat_and_stands_up_to_stretch() -> None:
    asked_seated, started, labels = _live("0.18.5")
    assert asked_seated == []  # nothing asked from the seat
    assert started == ["stretch_tall"]  # it stood up and stretched standing
    after = labels[labels.index("stretch_tall") :]
    assert after[-1] == "sit"  # and sat back down: the brain wanted to stay seated


def test_v5_keeps_its_seated_stretch() -> None:
    asked_seated, started, labels = _live("0.15.0")
    assert asked_seated == ["stretch_sit"] and started == ["stretch_sit"]
    assert "rise" not in labels  # no stand-up


def test_v8_does_not_stand_up_for_an_act_v5_shows_no_seated_move_for() -> None:
    asked_seated, started, labels = _live("0.18.5", behaviour="body_shake")
    assert asked_seated == [] and started == [] and set(labels) == {"sit"}
