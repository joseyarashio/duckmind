"""Speed by what the duck is doing (owner 2026-10-04: 「鴨子各項反應速率應該是有速率差別」).

A flight in panic (flee, retreat) runs at the top of the band at once, with no pauses and no
wandering, turning away in the tightest arc the body allows (an arc is never faster than 0.40);
an ordinary walk keeps to 0.35–0.45 by temperament; beside the thing it investigates it is at the
gait's slowest, and behind the person it follows it catches up; a startle whose threat has not gone
becomes the flight at once."""

from __future__ import annotations

import statistics

import pytest
from test_neuro_brain import _genome

from duckmind.neuro import brain as B
from duckmind.neuro import motor as M
from duckmind.neuro import select as S
from duckmind.neuro.contract import ARC_VX_MIN, NORMAL_VX_MAX

STANDING = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)


def _run(seconds: float, **kw: object) -> list[M.LocoOut]:
    params = M.LocoParams()
    state = M.LocoState(rng=3)
    outs = []
    for k in range(int(seconds * 50)):
        state, out = M.loco_step(params, state, M.LocoInput(**kw), k * 20)  # type: ignore[arg-type]
        outs.append(out)
    return outs


def test_a_panic_goes_from_the_first_tick_and_never_pauses() -> None:
    outs = _run(30.0, go=1.0, panic=True, wander_scale=0.0)
    assert all(o.walking for o in outs[1:])
    assert all(o.vx > 0.0 for o in outs[1:])


def test_a_panic_runs_at_the_top_of_the_band_but_never_at_one_speed() -> None:
    speeds = [o.vx for o in _run(60.0, go=1.0, panic=True, wander_scale=0.0) if o.vx > 0.0]
    assert min(speeds) >= M.PANIC_VX_MIN - 1e-9 and max(speeds) <= NORMAL_VX_MAX + 1e-9
    assert statistics.median(speeds) >= 0.52
    assert len({round(v, 3) for v in speeds}) > 20


def test_a_panic_arc_is_never_faster_than_the_body_allows() -> None:
    outs = _run(20.0, go=1.0, panic=True, wander_scale=0.0, turn=1.0)
    arcs = [o for o in outs if o.vx > 0.0]
    assert arcs and all(abs(o.vyaw) > 0.3 for o in arcs)
    assert all(o.vx <= 0.40 + 1e-9 for o in arcs)


def test_an_ordinary_walk_still_pauses_and_stays_in_the_ordinary_band() -> None:
    outs = _run(120.0, go=0.8, vx_cap=B.PACE_FAST)
    assert any(not o.walking for o in outs)
    speeds = [o.vx for o in outs if o.vx > 0.0]
    assert max(speeds) <= B.PACE_FAST + 1e-9 and min(speeds) >= ARC_VX_MIN - 1e-9


def test_the_ordinary_paces_in_the_table_are_not_the_top_of_the_band() -> None:
    for beh in ("wander", "investigate", "follow", "return_dock"):
        assert B.LOCO[beh][0] == B.PACE_FAST < NORMAL_VX_MAX
    assert B.LOCO["forage"][0] <= B.PACE_FAST
    assert B.LOCO["flee"][0] == B.LOCO["retreat"][0] == NORMAL_VX_MAX
    assert {"flee", "retreat"} == set(M.PANIC_ACTS)


def _brain(preset: str) -> B.NeuroBrain:
    return B.NeuroBrain(_genome(preset))


@pytest.mark.parametrize("bold,lazy", [("bold_explorer", "lazy_dreamer")])
def test_the_bold_walk_a_little_faster_than_the_lazy(bold: str, lazy: str) -> None:
    fast = _brain(bold)._pace("wander", B.PACE_FAST, None)
    slow = _brain(lazy)._pace("wander", B.PACE_FAST, None)
    assert B.PACE_SLOW <= slow < fast <= B.PACE_FAST


def test_beside_the_thing_it_investigates_it_is_at_the_slowest() -> None:
    brain = _brain("bold_explorer")
    assert brain._pace("investigate", B.PACE_FAST, 0.2) == ARC_VX_MIN
    assert brain._pace("investigate", B.PACE_FAST, 1.5) > ARC_VX_MIN


def test_behind_the_person_it_follows_it_catches_up() -> None:
    brain = _brain("bold_explorer")
    near = brain._pace("follow", B.PACE_FAST, 0.6)
    far = brain._pace("follow", B.PACE_FAST, 3.0)
    assert far > near
    assert far == pytest.approx(NORMAL_VX_MAX)


def test_a_committed_switch_takes_the_body_and_keeps_it() -> None:
    sel = S.Selector(_genome("bold_explorer"), seed=1)
    feat = S.empty_features()
    feat[S.F["restless"]] = 0.7
    feat[S.F["energy"]] = 0.9
    for _ in range(100):
        sel.step(0.05, feat, STANDING, False, 0.0)
    assert sel.names[sel.current] != "flee"
    sel.commit_next("flee")
    name, new = sel.step(0.05, feat, STANDING, False, 0.0)
    assert (name, new) == ("flee", True)
    for _ in range(40):  # two seconds on: still the flight
        assert sel.step(0.05, feat, STANDING, False, 0.0)[0] == "flee"
    # never into a walk it cannot make
    sel.commit_next("wander")
    cannot = S.Context(sitting=False, standing=True, asleep=False, can_walk=False, wakefulness=1.0)
    assert sel.step(0.05, feat, cannot, False, 0.0)[0] != "wander"


# ── a strong scare is a flight for every temperament (owner 2026-10-05) ──


@pytest.mark.parametrize(
    "preset", ["bold_explorer", "shy_snuggler", "lazy_dreamer", "social_songbird"]
)
def test_a_strong_scare_is_a_flight_for_every_temperament_for_a_while(preset: str) -> None:
    b = _brain(preset)
    assert b.drives.fear == 0.0
    assert not b._fright_persists(10.0)  # (calm: nothing to run from)
    b._strong_scare_s = 10.0
    assert b._fright_persists(10.0 + B.STARTLE_FLIGHT_S - 0.1)
    assert not b._fright_persists(10.0 + B.STARTLE_FLIGHT_S + 0.1)


def test_the_bold_still_stand_their_ground_to_a_mild_scare() -> None:
    bold = _brain("bold_explorer")
    shy = _brain("shy_snuggler")
    for b in (bold, shy):
        b.drives.fear = 0.40  # a medium fright, no strong scare
    assert not bold._fright_persists(5.0) and shy._fright_persists(5.0)


def test_things_in_view_do_not_habituate_a_hand_rushing_at_the_face() -> None:
    from duckmind.neuro.habituation import Habituation

    hab = Habituation(_genome("bold_explorer").p)
    for k in range(12):  # the explorer keeps looking at the things in its home
        hab.present("visual:thing", 0.4, 10.0 + 4.0 * k, "orient")
    assert hab.level("visual:thing", 58.0) < 0.5
    assert hab.level("visual:looming", 58.0) == 1.0
    assert hab.level("visual:person", 58.0) < 1.0  # (the other sights still generalise)


@pytest.mark.parametrize(
    "preset", ["bold_explorer", "shy_snuggler", "lazy_dreamer", "social_songbird"]
)
def test_a_bang_through_the_startle_floor_is_a_flight_even_when_the_home_has_habituated_it(
    preset: str,
) -> None:
    """The sharp sounds of its home habituate the group a little (a fresh boot reads ~0.78, under
    STRONG_SCARE / raw 1.0): the startle floor's startle is a strong scare itself, so the first
    bang is a jump and a flight for the bold too. A bang it is well used to (level 0.4) is not a flight for anyone."""
    from duckmind.neuro import contract as C
    from duckmind.neuro.habituation import GROUPS

    def fright_after_bang(level: float) -> tuple[bool, bool]:
        brain = B.NeuroBrain(_genome(preset), seed=3)
        t_ms = 0
        while t_ms < 4000:
            brain.step(t_ms, C.SensorFrame(t_ms=t_ms, body=C.BodySense(posture="stand")))
            t_ms += 50
        for cls in GROUPS["sharp_sound"]:
            brain.hab._h[cls] = (level, (t_ms - brain._t0) / 1000.0)
        brain.hab._l = (level, (t_ms - brain._t0) / 1000.0)  # (used to loud sounds too)
        bang = C.SoundSense(level_db=94.0, onset=1.0, kind="bang")
        startled = False
        while t_ms < 4000 + 3000:
            frame = C.SensorFrame(
                t_ms=t_ms,
                body=C.BodySense(posture="stand"),
                sound=bang if t_ms == 4000 else C.SoundSense(),
            )
            _, tel = brain.step(t_ms, frame)
            startled = startled or any(e.reflex == "startle" for e in tel.reflexes)
            t_ms += 50
        return startled, brain._fright_persists((t_ms - brain._t0) / 1000.0 - 1.0)

    assert fright_after_bang(0.7) == (True, True)
    assert fright_after_bang(0.4)[1] is False


@pytest.mark.parametrize(
    "preset", ["bold_explorer", "shy_snuggler", "lazy_dreamer", "social_songbird"]
)
def test_a_flight_after_a_startle_is_not_lost_while_a_trained_skill_still_has_the_body(
    preset: str,
) -> None:
    """Bug 0 (2026-10-09). With the trained skills installed (309377c) a startle and its freeze are robotd skills; the
    label of a skill is not 'stand' (the body is crouched, then in the one-rise release), so for the tick the reflex
    layer let go the brain read posture 'unknown', skipped the flight and forgot the startle: the explorer startled at
    the bang (floor true) and then wandered on (live fresh boots: 4 of 12 on a707a50, 0 of 13 on eb08fe8). The flight is
    now kept waiting until the body is standing again (within STARTLE_FLIGHT_S of the hold's end)."""
    from duckmind.neuro import contract as C

    brain = B.NeuroBrain(_genome(preset), seed=3)
    t_ms = 0
    while t_ms < 4000:
        brain.step(t_ms, C.SensorFrame(t_ms=t_ms, body=C.BodySense(posture="stand")))
        t_ms += 50
    bang = C.SoundSense(level_db=94.0, onset=1.0, kind="bang")
    skill_until = (
        4000 + 4200
    )  # (startle 1.3 s + a 2 s held crouch + startle_release's rise ~0.9 s from the bang)
    behaviours: list[str] = []
    startled = False
    while t_ms < 4000 + 9000:
        posture = (
            "unknown" if 4000 <= t_ms < skill_until else "stand"
        )  # (a skill's label is no posture it knows)
        frame = C.SensorFrame(
            t_ms=t_ms,
            body=C.BodySense(posture=posture),
            sound=bang if t_ms == 4000 else C.SoundSense(),
        )
        _, tel = brain.step(t_ms, frame)
        if any(e.reflex == "startle" for e in tel.reflexes):
            startled = True
        if not behaviours or behaviours[-1] != tel.behaviour:
            behaviours.append(tel.behaviour)
        t_ms += 50
    assert startled
    assert "flee" in behaviours, behaviours


def test_quiet_knocks_of_a_thing_it_pushes_do_not_take_the_bang_away_from_the_startle_floor() -> (
    None
):
    """Bug 0 (2026-10-09). The startle floor asks whether the duck is used to LOUD sudden sounds (its own level,
    habituation.LOUD_KEY), not only the bang class's level: 30 knocks of raw 0.6-0.7 (a tissue box pushed along, 64 dB
    at the head, a third of them named "bang" by the ear) left the bang class at 0.15 and the first page bang was no
    startle (live, 4 of 8 fresh boots with the 20 ms servo delay). A run of loud claps still makes it used to bangs."""
    from duckmind.neuro import contract as C
    from duckmind.neuro.habituation import LOUD_KEY, Habituation

    hab = Habituation(_genome("bold_explorer").p)
    for k in range(30):
        hab.present("sound:bang", 0.7, 10.0 + 2.0 * k, None)  # (the knock the ear took for a bang)
        hab.present("sound:door", 0.6, 11.0 + 2.0 * k, None)
    assert hab.level("sound:bang", 70.0) < 0.3  # (the class is used to those)
    assert hab.level(LOUD_KEY, 70.0) == 1.0  # (the loud sounds are not)

    claps = Habituation(_genome("bold_explorer").p)
    for k in range(8):
        claps.present("sound:clap", 0.89, 10.0 + 3.0 * k, "startle")
    assert (
        claps.level(LOUD_KEY, 40.0) < 0.6
    )  # (a duck that has heard loud claps is used to loud sounds)

    # the whole brain: the knocks first, then the page's bang
    brain = B.NeuroBrain(_genome("bold_explorer"), seed=3)
    t_ms = 0
    while t_ms < 4000:
        brain.step(t_ms, C.SensorFrame(t_ms=t_ms, body=C.BodySense(posture="stand")))
        t_ms += 50
    now_s = (t_ms - brain._t0) / 1000.0
    for k in range(30):
        brain.hab.present("sound:bang", 0.7, now_s + 0.1 * k, None)
    bang = C.SoundSense(level_db=94.0, onset=1.0, kind="bang")
    startled = False
    beh: list[str] = []
    while t_ms < 4000 + 6000:
        frame = C.SensorFrame(
            t_ms=t_ms,
            body=C.BodySense(posture="stand"),
            sound=bang if t_ms == 4000 else C.SoundSense(),
        )
        _, tel = brain.step(t_ms, frame)
        startled = startled or any(e.reflex == "startle" for e in tel.reflexes)
        if not beh or beh[-1] != tel.behaviour:
            beh.append(tel.behaviour)
        t_ms += 50
    assert startled and "flee" in beh, (startled, beh)
