"""robotd 0.16.0+ runs one move at a time (pollen-robotics/microduck f93ecf7): a skill requested while
another runs is refused, so ``skill_abort`` and the releases no longer cut in. Sent anyway, they start as a
fresh skill right AFTER the run ended — the walk network reset under a zero command mid-stride (seen on
0.18.5: ``skill started skill=skill_abort`` after every quick turn and flight). On such a robotd the
hand-backs end a held run by not refreshing it (UPSTREAM-0185-2026-11-05)."""

from __future__ import annotations

from test_neuro_bodylink import DT, moves, state

from duckmind.duckbody.robotd_client import RobotdClient, daemon_before
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import guard as G

NAMES = [*G.PANIC_ENTRIES, *G.QUICK_TURN_ENTRIES, G.ABORT, G.RELEASE, "startle_3", "freeze_3"]


def view(label: str, *, twist: float = 0.4, tilt: float = 2.0) -> G.BodyView:
    return G.BodyView(label, twist=twist, vx=twist, tilt_deg=tilt)


def _do(out) -> list[str]:
    return [i.params["skill"] for i in out.intents if i.method == "robot.do"]


def _flight(interrupt: bool) -> list[list[str]]:
    """A flight, then the brain stops fleeing: what the link sends while robotd still runs the
    panic skill (its window not over) and once robotd's walk has it back."""
    link = BodyLink()
    link.skills = G.SkillGuard(NAMES, interrupt=interrupt)
    t = 0.0
    for _ in range(40):
        t += DT
        link.tick(state(t, policy="walk"), BodyRequest(vx=0.4))
    flee = BodyRequest(vx=0.5, panic=True)
    for k in range(60):
        t += DT
        link.tick(state(t, policy="walk" if k == 0 else "panic_run"), flee)
    sent = []
    for label in ("panic_run", "panic_run", "panic_run", "walk", "walk"):
        t += DT
        out = link.tick(state(t, policy=label), BodyRequest(vx=0.4))
        sent.append(_do(out))
        if label == "walk":
            assert moves(out.intents)  # the walk goes on (the hand-back coast)
    return sent


def test_a_flight_is_handed_back_by_its_window_on_a_one_move_robotd() -> None:
    assert all(not s for s in _flight(interrupt=False))  # neither the abort nor a refresh
    assert _flight(interrupt=True)[0] == [G.ABORT]  # the older robotd: cut in at once, as before


def test_the_end_entries_are_not_sent_while_a_run_holds_the_body() -> None:
    g = G.SkillGuard(NAMES, interrupt=False)
    for end, running in ((G.ABORT, "quick_turn_left"), (G.RELEASE, "freeze_3")):
        d = g.request(end, view(running), 10.0)
        assert not d.ok and d.skill is None  # nothing to send: the run ends by itself
    assert g.request(G.ABORT, view("walk"), 10.1).reason == "nothing"


def test_a_freeze_does_not_take_over_a_running_startle() -> None:
    old, new = G.SkillGuard(NAMES), G.SkillGuard(NAMES, interrupt=False)
    assert old.request("freeze_3", view("startle_3"), 5.0).reason == "continue"
    assert new.request("freeze_3", view("startle_3"), 5.0).reason == "busy"


def test_a_tipped_held_run_is_no_longer_refreshed() -> None:
    g = G.SkillGuard(NAMES, interrupt=False)
    assert g.request("panic_run", view("panic_run"), 1.0).reason == "hold"
    assert g.watch(view("panic_run", tilt=40.0), 1.02) is None  # nothing to send …
    assert (
        g.request("panic_run", view("panic_run", tilt=40.0), 1.04).reason == "tilt"
    )  # … no refresh
    assert g.watch(view("walk"), 1.5) is None
    assert (
        g.request("panic_run", view("panic_run"), 30.0).reason == "hold"
    )  # a later run: refreshed
    assert G.SkillGuard(NAMES).watch(view("panic_run", tilt=40.0), 1.02) == G.ABORT


def test_the_daemon_says_whether_skills_cut_in() -> None:
    client = RobotdClient(None, None, None)  # type: ignore[arg-type]
    for version, cuts_in in (("0.15.0", True), ("0.15.3", True), ("0.16.0", False),
                             ("0.18.5", False), ("0.19.0-rc.1", False), (None, False), ("v5", False)):  # fmt: skip
        client.hello = {"api_version": 49, "daemon_version": version}
        assert client.skills_interrupt is cuts_in, version
    assert daemon_before("0.15.9+ci", (0, 16, 0))
