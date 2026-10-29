"""Live voice: the speakers the owner picks on the page. A family role (主人, 伴侶, 小孩, 長輩) is a
known person to the duck, the body that stands for them is introduced by the role's name; a guest
is a stranger. A person's answer claims the body only when the person is known (or nobody is
addressed); a stranger's answer goes through the plain competition (LV4's reserve). No network."""

from __future__ import annotations

from typing import Any

import pytest
from live_bridge_rig import frame, make_brain, person_sight, run_brain, tag, vocalize

from duckmind.live.bridge import FAMILY_HINT_S, LiveBridge
from duckmind.simweb.live_voice import LiveVoice

pytestmark = pytest.mark.usefixtures("no_network")


def _bridge() -> tuple[Any, Any, LiveBridge]:
    brain, clock = make_brain()
    bridge = LiveBridge(clock=clock)
    brain.live = bridge
    return brain, clock, bridge


def _see(brain: Any, clock: Any, seconds: float) -> None:
    run_brain(
        brain,
        clock,
        seconds,
        frames=lambda t: frame(int(round(t * 1000)), person_sight(track=1, range_m=1.2)),
    )


def test_a_family_speaker_makes_the_body_in_view_a_known_person() -> None:
    brain, clock, bridge = _bridge()
    bridge.family_speaks("主人")
    _see(brain, clock, 1.0)
    assert list(brain.learn.person_names().values()) == ["主人"]
    assert bridge.counts["family_introduced"] == 1


def test_the_same_role_spoken_again_for_the_same_person_is_one_person() -> None:
    """The family-speaker introduction twice (the eye re-issued the track, no look to say it is the same
    one): one 伴侶 in the people list, not two."""
    brain, clock, bridge = _bridge()
    bridge.family_speaks("伴侶")
    _see(brain, clock, 1.0)
    run_brain(brain, clock, 20.0)  # (they walk out of view; the eye forgets the track)
    bridge.family_speaks("伴侶")
    run_brain(
        brain,
        clock,
        1.0,
        frames=lambda t: frame(int(round(t * 1000)), person_sight(track=7, range_m=1.2)),
    )
    assert list(brain.learn.person_names().values()) == ["伴侶"]
    assert sum(1 for p in brain.learn.people.values() if p.name == "伴侶") == 1


def test_nobody_is_introduced_without_a_family_speaker() -> None:
    brain, clock, bridge = _bridge()
    _see(brain, clock, 1.0)
    assert brain.learn.person_names() == {}


def test_the_role_is_held_only_a_while() -> None:
    brain, clock, bridge = _bridge()
    bridge.family_speaks("主人")
    run_brain(brain, clock, FAMILY_HINT_S + 2.0)  # (nobody in view)
    _see(brain, clock, 1.0)
    assert brain.learn.person_names() == {}


def test_the_page_gives_a_family_role_to_the_bridge_and_never_a_guest() -> None:
    told: list[str] = []

    class Stub:
        def family_speaks(self, name: str) -> None:
            told.append(name)

    voice = LiveVoice.__new__(LiveVoice)
    voice.bridge = Stub()  # type: ignore[assignment]
    voice.family_speaks("guest")
    voice.family_speaks("")
    assert told == []
    voice.family_speaks("partner")
    voice.family_speaks("owner")
    assert len(told) == 2 and all(told)


def _claims(brain: Any) -> list[dict[str, float]]:
    seen: list[dict[str, float]] = []
    original = brain.sel.set_extra

    def spy(extras: dict[str, float], *, claims: dict[str, float] | None = None) -> Any:
        seen.append(dict(claims or {}))
        return original(extras, claims=claims)

    brain.sel.set_extra = spy
    return seen


@pytest.mark.parametrize("named", [True, False])
def test_only_a_known_persons_reply_claims_the_body(named: bool) -> None:
    from duckmind.live.gate import Verdict, VocalTag

    brain, clock, bridge = _bridge()
    _see(brain, clock, 1.0)
    (pid,) = [p.pid for p in bridge.perception.people]
    if named:
        brain.learn.bind_name(pid, "媽媽", "媽媽", clock.t)
    seen = _claims(brain)
    call = vocalize([tag("greet", 0.8)])
    verdict = Verdict(call, True, None, tags=(VocalTag("greet", 0.8, 0),))
    bridge._vocalize(call, verdict, clock.t, brain)
    _see(brain, clock, 0.3)
    assert any(seen) is named
