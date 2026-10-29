"""Live voice, LV4: the two new drives in the bridge, on a real integrated brain. 親近 (bond) is the
learner's, per person: a startle by the person it sees lowers it, introducing them and being talked to
raise it, and it makes the reply, the wish to start a conversation and the wish to go to them stronger.
想說話 (talk) shapes how much, how soon and for how long it answers, starts a conversation of its own,
and is told to the cloud with the bond. Nothing is voiced while it is frightened; nothing changes when
nobody is there. No network, no physics."""

from __future__ import annotations

from typing import Any

import pytest
from live_bridge_rig import (
    MOM_LOOK,
    frame,
    make_brain,
    make_call,
    person_sight,
    run_brain,
    tag,
    vocalize,
)

from duckmind.live import bridge as B
from duckmind.live import soul as S
from duckmind.live.drives import FEAR_MUTE, DriveParams
from duckmind.neuro.synth import SynthWorld, iterate

pytestmark = pytest.mark.usefixtures("no_network")

SNUGGLER = (1728, 20)
EXPLORER = (4107, 60)
HATCHLING = (3341, 7)


def rig(
    params: DriveParams | None = None, who: tuple[int, int] = SNUGGLER, **config: Any
) -> tuple[Any, B.LiveBridge, Any]:
    brain, clock = make_brain(*who)
    cfg = B.BridgeConfig(drives=params or DriveParams.neutral(), **config)
    return brain, B.LiveBridge(None, clock=clock, config=cfg).attach(brain), clock


def with_person(
    *, kind: str | None = None, db: float = 30.0, hit: tuple[float, float] | None = None
) -> Any:
    """Frames with mom in front of the duck; a sound of ``kind`` between the two times of ``hit``."""

    def make(t: float) -> Any:
        loud = hit is not None and hit[0] - 0.026 <= t < hit[1]
        return frame(
            int(round(t * 1000)),
            person_sight(1, look=MOM_LOOK, bearing=0.0, range_m=1.4),
            sound_kind=kind if (loud or hit is None) and kind else None,
            level_db=db if (loud or hit is None) else 30.0,
        )

    return make


def pid_of_mom(brain: Any) -> int:
    (pid,) = brain.learn.people
    return int(pid)


# ── 親近 raises and lowers what the duck does for a person ─────────────────────────────────


def test_a_persons_bond_makes_the_reply_for_them_stronger_and_a_strangers_weaker() -> None:
    params = DriveParams.neutral().with_values(stranger_floor=0.2, bond_full=0.5)
    brain, bridge, clock = rig(params)
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    intent = B.Intent(1, "reply", 1, clock.t, 5.0, ("greet",), level=0.6, addressee=pid, known=True)
    stranger = bridge._weight(intent, brain)
    brain.learn.people[pid].bond = 0.25
    halfway = bridge._weight(intent, brain)
    brain.learn.people[pid].bond = 0.8
    close = bridge._weight(intent, brain)
    assert stranger < halfway < close
    assert close == pytest.approx(stranger / 0.2, rel=1e-2)
    nobody = B.Intent(2, "reply", 1, clock.t, 5.0, ("greet",), level=0.6)
    assert bridge._weight(nobody, brain) == pytest.approx(close, rel=1e-2)  # (a voice, no face)


def test_a_startle_when_a_person_is_in_view_lowers_their_bond_and_the_reply_for_them() -> None:
    params = DriveParams.neutral().with_values(bond_l_startle=0.5, bond_full=0.5)
    brain, bridge, clock = rig(params, EXPLORER)
    run_brain(brain, clock, 4.0, frames=with_person())
    pid = pid_of_mom(brain)
    brain.learn.people[pid].bond = 0.6
    intent = B.Intent(1, "reply", 1, clock.t, 5.0, ("greet",), level=0.6, addressee=pid, known=True)
    before = bridge._weight(intent, brain)
    clap = with_person(kind="clap", db=88.0, hit=(clock.t + 1.0, clock.t + 1.04))
    cmds = run_brain(brain, clock, 3.0, frames=clap)
    assert any(c.reflex in B.STARTLE_REFLEXES for c in cmds)
    assert bridge.drives is not None and bridge.drives.counts["startled"] == 1
    assert brain.learn.people[pid].bond < 0.35
    brain.drives.fear = 0.0
    assert bridge._weight(intent, brain) < before


def test_a_startle_with_nobody_in_view_blames_nobody() -> None:
    brain, bridge, clock = rig(DriveParams.neutral(), SNUGGLER)
    run_brain(brain, clock, 3.0)
    t = clock.t

    def clap(x: float) -> Any:
        hit = abs(x - (t + 1.0)) < 0.026
        return frame(
            int(round(x * 1000)), sound_kind="bang" if hit else None, level_db=92.0 if hit else 30.0
        )

    cmds = run_brain(brain, clock, 3.0, frames=clap)
    assert any(c.reflex in B.STARTLE_REFLEXES for c in cmds)
    assert bridge.drives is not None and bridge.drives.counts["startled"] == 0


def test_being_introduced_gives_the_person_a_first_bond() -> None:
    params = DriveParams.neutral().with_values(bond_g_intro=0.3, bond_g_talk=0.0)
    brain, bridge, clock = rig(params)
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    assert brain.learn.people[pid].bond == 0.0
    bridge.submit(make_call("recognize", {"name": "媽媽"}, t=clock.t, seq=1))
    run_brain(brain, clock, 0.5, frames=with_person())
    assert brain.learn.people[pid].name == "媽媽"
    assert brain.learn.people[pid].bond == pytest.approx(0.3, abs=0.01)


def test_talk_heard_from_the_person_in_view_raises_their_bond() -> None:
    params = DriveParams.neutral().with_values(bond_g_talk=0.1)
    brain, bridge, clock = rig(params)
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    run_brain(brain, clock, 3.0, frames=with_person(kind="voice", db=62.0))
    assert brain.learn.people[pid].bond == pytest.approx(0.1, abs=0.01)
    assert bridge.drives is not None and bridge.drives.counts["talked_to"] == 1


def test_a_name_heard_and_a_friendly_proposal_make_it_warm_to_the_person() -> None:
    params = DriveParams.neutral().with_values(bond_g_name=0.2, bond_g_talk=0.0)
    brain, bridge, clock = rig(params)
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    run_brain(brain, clock, 1.0, frames=with_person(kind="name", db=64.0))
    bridge.submit(vocalize([tag("greet", 0.6, 0)], t=clock.t, seq=1))
    run_brain(brain, clock, 0.5, frames=with_person())
    assert brain.learn.people[pid].bond == pytest.approx(0.2, abs=0.01)
    assert bridge.drives is not None and bridge.drives.counts["called_warmly"] == 1


# ── 想說話 shapes the reply ────────────────────────────────────────────────────────────────


def _first_voiced_after(bridge: B.LiveBridge, since: float) -> float | None:
    times = [v["t"] for v in bridge.voiced if v["t"] >= since]
    return min(times) if times else None


def test_a_slow_duck_waits_before_its_first_tag() -> None:
    quick = DriveParams.neutral()
    slow = quick.with_values(hesitate_s=1.5)
    waits = []
    for params in (quick, slow):
        brain, bridge, clock = rig(params, reply_weight=30.0)
        run_brain(brain, clock, 20.0)
        t = clock.t
        bridge.submit(vocalize([tag("greet", 0.6, 0)], t=t, seq=1))
        run_brain(brain, clock, 8.0)
        first = _first_voiced_after(bridge, t)
        assert first is not None
        waits.append(first - t)
    # (the quick duck's reply is held back 0.45 s behind the duck's own greeting at that moment, so a
    # call of its own and the reply are never on top of each other (LV6, min_gap_s); the slow one is
    # due 1.5 s after the proposal: 1.0 s apart to the tick, so with float rounding)
    assert waits[0] < 1.0 and waits[1] >= 1.5 and waits[1] - waits[0] >= 1.0 - 1e-9


def test_a_reply_goes_on_to_all_its_tags_while_the_wish_lasts_and_stops_when_it_is_gone() -> None:
    three = [tag("greet", 0.7, 0), tag("chirp", 0.5, 700), tag("coo", 0.4, 1500)]
    said: dict[float, int] = {}
    for gate in (0.0, 0.95):
        brain, bridge, clock = rig(
            DriveParams.neutral().with_values(continue_wish=gate), reply_weight=30.0
        )
        run_brain(brain, clock, 20.0)
        bridge.submit(vocalize(three, t=clock.t, seq=1))
        run_brain(brain, clock, 10.0)
        said[gate] = len(bridge.voiced)
        if gate:
            assert bridge.reasons["brief"] == 1  # (it said so)
    assert said[0.0] == 3 and said[0.95] == 1


def test_nothing_is_voiced_while_frightened_however_strong_the_proposal() -> None:
    results = {}
    for fear in (FEAR_MUTE - 0.1, FEAR_MUTE + 0.05):
        brain, bridge, clock = rig(DriveParams.neutral(), reply_weight=60.0)

        def scared(_t: float, _cmd: Any, level: float = fear, drives: Any = brain.drives) -> None:
            drives.fear = level

        run_brain(brain, clock, 15.0, on_tick=scared)
        bridge.submit(vocalize([tag("greet", 0.9, 0)], t=clock.t, seq=1))
        run_brain(brain, clock, 8.0, on_tick=scared)
        results[fear] = len(bridge.voiced)
    low, high = sorted(results)
    assert results[low] >= 1 and results[high] == 0


def test_a_voiced_reply_to_speech_is_an_exchange_and_takes_a_part_of_the_wish() -> None:
    params = DriveParams.neutral().with_values(talk_sat_exchange=0.4)
    brain, bridge, clock = rig(params, reply_weight=30.0)
    run_brain(brain, clock, 15.0, frames=with_person())
    assert bridge.drives is not None
    bridge.drives.talk.value = 0.8
    run_brain(brain, clock, 1.5, frames=with_person(kind="voice", db=62.0))
    bridge.submit(vocalize([tag("greet", 0.6, 0)], t=clock.t, seq=1))
    run_brain(brain, clock, 3.0, frames=with_person())
    assert bridge.voiced and bridge.drives.counts["exchange"] == 1
    assert bridge.drives.talk.value < 0.8 * 0.7
    pid = pid_of_mom(brain)
    assert brain.learn.people[pid].exchanges == 1


# ── wishes of its own ──────────────────────────────────────────────────────────────────────


def test_a_duck_that_wants_to_talk_to_a_person_it_trusts_asks_for_a_call_out() -> None:
    params = DriveParams.neutral().with_values(init_gain=2.0, init_threshold=0.4, bond_full=0.5)
    brain, bridge, clock = rig(params)
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    brain.learn.people[pid].bond = 0.6
    assert bridge.drives is not None
    bridge.drives.talk.value = 0.95
    run_brain(brain, clock, 0.2, frames=with_person())
    assert bridge.last_extras.get("call_out", 0.0) > 0.5
    # …but not while a person is speaking, nor with nobody in view, nor with no wish
    bridge.drives.talk.value = 0.95
    run_brain(brain, clock, 0.2, frames=with_person(kind="voice", db=62.0))
    assert "call_out" not in bridge.last_extras
    bridge.drives.talk.value = 0.95
    run_brain(brain, clock, 0.2)
    assert "call_out" not in bridge.last_extras
    bridge.drives.talk.value = 0.1
    run_brain(brain, clock, 0.2, frames=with_person())
    assert "call_out" not in bridge.last_extras


def test_the_wish_to_start_a_conversation_shows_in_what_the_duck_does() -> None:
    calls = {}
    for gain in (0.0, 3.0):
        brain, bridge, clock = rig(
            DriveParams.neutral().with_values(init_gain=gain, init_threshold=0.3), who=HATCHLING
        )
        run_brain(brain, clock, 3.0, frames=with_person())
        brain.learn.people[pid_of_mom(brain)].bond = 0.6
        assert bridge.drives is not None

        def want(_t: float, _c: Any, d: Any = bridge.drives) -> None:
            d.talk.value = 0.95

        cmds = run_brain(brain, clock, 40.0, frames=with_person(), on_tick=want)
        calls[gain] = sum(1 for c in cmds if c.behaviour == "call_out")
    assert calls[3.0] > calls[0.0] + 10


def test_it_goes_toward_a_person_it_is_close_to_who_has_just_spoken_and_not_toward_others() -> None:
    params = DriveParams.neutral().with_values(approach_gain=0.4, bond_full=0.5)
    brain, bridge, clock = rig(params)
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    brain.learn.people[pid].bond = 0.6
    run_brain(brain, clock, 0.2, frames=with_person())
    assert "follow" not in bridge.last_extras  # (nobody has said anything to it)
    run_brain(brain, clock, 1.0, frames=with_person(kind="voice", db=62.0))
    run_brain(brain, clock, 0.2, frames=with_person())
    assert bridge.last_extras.get("follow", 0.0) == pytest.approx(0.4, rel=0.15)
    brain.learn.people[pid].bond = 0.0
    run_brain(brain, clock, 0.2, frames=with_person())
    assert "follow" not in bridge.last_extras  # (a stranger who spoke: no wish to go to them)


# ── what the cloud is told ─────────────────────────────────────────────────────────────────


def test_the_snapshot_tells_the_cloud_the_wish_to_talk_and_how_close_it_is_to_each_person() -> None:
    brain, bridge, clock = rig(DriveParams.neutral())
    run_brain(brain, clock, 3.0, frames=with_person())
    pid = pid_of_mom(brain)
    brain.learn.bind_name(pid, "媽媽", "媽媽", 1.0)
    brain.learn.people[pid].bond = 0.7
    assert bridge.drives is not None
    bridge.drives.talk.value = 0.6
    run_brain(brain, clock, 0.2, frames=with_person())
    snap = bridge.snapshot(brain)
    assert "talk" in snap.drives and "bond" in snap.drives
    assert snap.drives["bond"] == pytest.approx(0.7, abs=0.01)
    assert snap.bonds["媽媽"] == pytest.approx(0.7, abs=0.01) and snap.known_people == ("媽媽",)
    text = snap.text()
    assert "想說話" in text and "親近" in text and "媽媽（親近 7）" in text
    assert "媽媽" in text.split("眼睛現在看到：")[1]


def test_the_snapshot_goes_to_the_session_at_its_own_pace_when_asked_to() -> None:
    class Session:
        enabled = True
        state = "online"

        def __init__(self, clock: Any) -> None:
            self.clock = clock
            self.pushed: list[Any] = []

        def take_calls(self, limit: int | None = None) -> list[Any]:
            return []

        def set_context(self, context: Any) -> None:
            self.pushed.append(context)

    for every, expect in ((None, 0), (5.0, 4)):
        brain, clock = make_brain(*SNUGGLER)
        session = Session(clock)
        bridge = B.LiveBridge(
            session, clock=clock, config=B.BridgeConfig(context_every_s=every)
        ).attach(brain)
        run_brain(brain, clock, 20.0)
        assert len(session.pushed) in (expect, expect + 1) if expect else not session.pushed
        assert all(isinstance(c, S.ContextSnapshot) for c in session.pushed)
        assert bridge.faults == 0


def test_the_bridge_view_carries_the_drives() -> None:
    brain, bridge, clock = rig(DriveParams.neutral())
    run_brain(brain, clock, 2.0, frames=with_person())
    view = bridge.view()
    assert view["drives"]["people"][0]["bond"] == 0.0
    assert view["drives"]["talk"] == pytest.approx(view["talk"], abs=0.01)


# ── nothing changes when nobody is there ───────────────────────────────────────────────────


@pytest.mark.parametrize("who", [EXPLORER, SNUGGLER, HATCHLING])
def test_with_no_family_in_the_world_the_bridge_changes_nothing(who: tuple[int, int]) -> None:
    from duckmind.neuro.genome import Genome
    from duckmind.neuro.integration import make_brain as build

    def trace(attach: bool) -> list[tuple[Any, ...]]:
        brain = build(Genome.from_seed(who[0], gd=who[1]), seed=3)
        if attach:
            B.LiveBridge(None, config=B.BridgeConfig()).attach(brain)
        world = SynthWorld(seed=5, family=False)
        world.clap(30_000)
        out = []
        for cmd, _tel in iterate(brain, world, 120.0, hz=10.0):
            out.append(
                (cmd.behaviour, cmd.bout, cmd.vx, cmd.vyaw, cmd.vocal, cmd.vocal_seq, cmd.reflex)
            )
        return out

    # (the bridge's own clock is the wall's: nothing it does depends on it when nobody is there)
    assert trace(False) == trace(True)
