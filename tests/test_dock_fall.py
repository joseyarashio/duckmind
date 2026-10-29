"""2026-10-11 the fall at the dock (docs/design/neural-duck/DOCK-FALL-2026-10-11.md): no whole-body skill starts from a reverse, and the dock's side
guides are low enough that a fallen duck is not propped on them."""

from __future__ import annotations

from test_neuro_bodylink import DT, state

from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import guard as G
from duckmind.world import dock

STARTLE = "startle_2"
HOP = "hop_back"


def _reversing_link(names: list[str]) -> tuple[BodyLink, float]:
    link = BodyLink()
    link.skills = G.SkillGuard(names)
    t = 0.0
    for _ in range(40):
        t += DT
        link.tick(state(t, policy="walk"), BodyRequest(vx=-0.4, docking=True, posture="stand"))
    assert link.streamer.last == (-0.40, 0.0)
    return link, t


def test_no_whole_body_skill_starts_from_a_reverse() -> None:
    link, t = _reversing_link([STARTLE, G.ABORT])
    # the docking request ends (the brain's startle) while the last reverse is still the last twist sent
    out = link.tick(state(t + DT, policy="walk"), BodyRequest(skill=STARTLE, posture="stand"))
    assert not [i for i in out.intents if i.method == "robot.do"]
    assert any(e["kind"] == "skill.refused" and e["why"] == "docked" for e in out.events)
    # standing still for two seconds later the same scare gets its startle
    t += DT
    for _ in range(100):
        t += DT
        link.tick(state(t, policy="stand"), BodyRequest(posture="stand"))
    out = link.tick(state(t + DT, policy="stand"), BodyRequest(skill=STARTLE, posture="stand"))
    assert [i.params["skill"] for i in out.intents if i.method == "robot.do"] == [STARTLE]


def test_a_running_skill_is_not_a_reverse_to_the_guard() -> None:
    """The hop-back skill steps backwards by design: a skill that robotd already runs (its label is a table entry) is handled as
    before, whatever the last twist sent was."""
    link, t = _reversing_link([STARTLE, HOP, G.ABORT])
    out = link.tick(state(t + DT, policy=HOP), BodyRequest(posture="stand"))
    assert not any(e["kind"] == "skill.refused" for e in out.events)


def test_the_docks_side_guides_do_not_prop_a_fallen_duck() -> None:
    """Live, 80 mm walls held a fallen duck at 45-95 degrees for good in 19 % of 128 laid-down trials; low ones almost never (the note)."""
    for style in dock.STYLES:
        g = dock.shape(style)
        assert (
            17 <= g.guide_height_mm <= 40
        )  # above the thighs' and hips' lowest (>= 17 mm), well under the 80 mm that propped it
