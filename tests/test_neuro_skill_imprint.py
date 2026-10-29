"""a:social — 跟著媽媽走: who-is-who from its own senses, imprinting, the need to stay near."""

from __future__ import annotations

import math
import random

import pytest

from duckmind.neuro import skill_imprint as I
from duckmind.neuro import skill_social_brain as SB
from duckmind.neuro.genome import Genome
from duckmind.world import neuro_social_synth as W
from duckmind.world.neuro_social_senses import LOOKS, observe_look


def _imprint(stage: str = "hatchling", soc: float = 0.9, bold: float = 0.5) -> I.Imprint:
    young = {"hatchling": 1.0, "duckling": 0.6, "juvenile": 0.25, "adult": 0.0}[stage]
    return I.Imprint(stage=stage, boldness=bold, sociability=soc, young=young)


def _seen(track: int, who: str, bearing: float, rng_m: float, rng: random.Random) -> I.PersonSeen:
    look = observe_look(who, "bright", rng_m, rng)
    return I.PersonSeen(track, bearing, rng_m, 0.0, look)


def test_the_same_person_stays_one_and_two_people_stay_two() -> None:
    rng = random.Random(3)
    people = I.Individuals()
    # (new eye tracks each time: known again by the look alone; a pass without a look — 5 % in
    # bright light — cannot tell who it was: None)
    ids_child = {people.assign(float(k), _seen(10 + k, "child", 0.0, 1.6, rng)) for k in range(30)}
    ids_owner = {people.assign(float(k), _seen(100 + k, "owner", 0.0, 1.6, rng)) for k in range(30)}
    ids_child.discard(None)
    ids_owner.discard(None)
    assert len(ids_child) == 1 and len(ids_owner) == 1
    assert ids_child != ids_owner
    assert len(people.people) == 2


def test_an_eye_track_keeps_its_person_without_a_look() -> None:
    rng = random.Random(4)
    people = I.Individuals()
    pid = people.assign(0.0, _seen(7, "elder", 0.2, 1.4, rng))
    assert people.assign(0.5, I.PersonSeen(7, 0.2, 1.4, 0.0, None)) == pid
    assert people.assign(0.5, I.PersonSeen(8, 0.2, 1.4, 0.0, None)) is None  # unknown track


def test_plasticity_follows_the_sensitive_period_and_sociability() -> None:
    assert I.plasticity("hatchling", 0.9) > I.plasticity("duckling", 0.9)
    assert I.plasticity("duckling", 0.9) > I.plasticity("juvenile", 0.9)
    assert I.plasticity("juvenile", 0.9) > I.plasticity("adult", 0.9)
    assert I.plasticity("hatchling", 0.9) > 2.0 * I.plasticity("hatchling", 0.1)


def _care_session(imp: I.Imprint, who: str, seconds: float, rng: random.Random) -> None:
    """Someone stands 0.8 m in front and pats it (a gentle touch, 3 s in every 9)."""
    odom = (0.0, 0.0, 0.0)
    t = 0.0
    while t < seconds:
        persons = [_seen(50, who, 0.05, 0.8, rng)] if int(t * 10) % 5 == 0 else []
        touching = (t % 9.0) < 3.0
        care = I.CareCues(touch_gentle=0.8 if touching else 0.0)
        imp.observe(t, 0.1, odom, persons, care, fresh=bool(persons))
        t += 0.1


def test_a_hatchling_imprints_on_its_carer_an_adult_does_not() -> None:
    young, old = _imprint("hatchling", 0.9), _imprint("adult", 0.1)
    _care_session(young, "child", 60.0, random.Random(1))
    _care_session(old, "child", 60.0, random.Random(1))
    assert young.mom is not None
    assert young.people.people[young.mom].bond > 0.6
    assert old.mom is None
    assert max(p.bond for p in old.people.people.values()) < I.MOM_MIN


def test_bonding_is_winner_take_most() -> None:
    imp = _imprint("hatchling", 0.9)
    rng = random.Random(2)
    _care_session(imp, "child", 60.0, rng)
    first = imp.mom
    t0 = 100.0
    for k in range(600):  # the owner then gives the same care
        t = t0 + 0.1 * k
        persons = [_seen(90, "owner", 0.05, 0.8, rng)] if k % 5 == 0 else []
        care = I.CareCues(touch_gentle=0.8 if (k % 90) < 30 else 0.0)
        imp.observe(t, 0.1, (0.0, 0.0, 0.0), persons, care, fresh=bool(persons))
    owner = [p for p in imp.people.people.values() if p.pid != first]
    assert imp.mom == first
    assert owner and owner[0].bond < 0.5 * imp.people.people[first].bond


def test_the_need_to_be_near_her_and_settling_at_her_feet() -> None:
    imp = _imprint("hatchling", 0.9, bold=0.2)
    _care_session(imp, "child", 60.0, random.Random(5))
    assert imp.mom is not None
    rng = random.Random(6)
    t = 100.0
    # she stands still 2.5 m away: far
    for _ in range(40):
        imp.observe(t, 0.1, (0.0, 0.0, 0.0), [_seen(60, "child", 0.1, 2.5, rng)], I.CareCues())
        t += 0.1
    far = imp.features((0.0, 0.0, 0.0), 0.0)
    assert far["mom_far"] > 0.5 and far["mom_near_still"] == 0.0
    # it has come within 0.7 m and she has not moved for a while: settle
    for _ in range(40):
        imp.observe(t, 0.1, (1.8, 0.0, 0.0), [_seen(60, "child", 0.1, 0.7, rng)], I.CareCues())
        t += 0.1
    near = imp.features((1.8, 0.0, 0.0), 0.0)
    assert near["mom_far"] == 0.0 and near["mom_near_still"] > 0.5
    goal = imp.goal((1.8, 0.0, 0.0))
    assert goal is not None and goal[1] == pytest.approx(0.7, abs=0.2)


def test_object_permanence_it_looks_where_she_was_and_she_is_gone() -> None:
    imp = _imprint("hatchling", 0.9)
    _care_session(imp, "child", 60.0, random.Random(7))
    view = imp.mom_view((0.0, 0.0, 0.0))
    assert view is not None and view.believed
    t = 70.0
    for _ in range(50):  # looking straight at where she stood, with a still head: nobody
        imp._t_s = t
        imp.observe(t, 0.1, (0.0, 0.0, 0.0), [], I.CareCues(), fresh=True)
        imp.check_absent(0.1, (0.0, 0.0, 0.0), head_yaw=0.05, can_see=True)
        t += 0.1
    view = imp.mom_view((0.0, 0.0, 0.0))
    assert view is not None and not view.believed
    assert "not_there" in " ".join(imp.events)


def test_a_touch_belongs_to_the_person_beside_it_not_one_across_the_room() -> None:
    imp = _imprint("hatchling", 0.9)
    rng = random.Random(8)
    odom = (0.0, 0.0, 0.0)
    imp.observe(0.0, 0.1, odom, [_seen(1, "child", 0.0, 0.8, rng)], I.CareCues())
    imp.observe(0.5, 0.1, odom, [_seen(2, "owner", 1.0, 3.0, rng)], I.CareCues())
    for k in range(30):  # it looks away (nobody in view) and is patted
        imp.observe(1.0 + 0.1 * k, 0.1, odom, [], I.CareCues(touch_gentle=0.8), fresh=True)
    bonds = {
        I.look_distance(p.mean, LOOKS["child"]) < 1.0: p.bond for p in imp.people.people.values()
    }
    assert bonds[True] > 0.1 and bonds.get(False, 0.0) == 0.0


@pytest.fixture(scope="module")
def social() -> object:
    with SB.registered():
        yield SB


def _synthetic_follow(
    genome: Genome, seed: int
) -> tuple[SB.SocialBrain, W.SocialSynth, list[float]]:
    brain = SB.SocialBrain(genome, seed=seed)
    world = W.SocialSynth(seed=seed)
    world.care(10_000, "child", 100)
    world.walk(30_000, "owner", (1.0, 1.0))
    world.leave(60_000, "owner")
    world.walk(150_000, "child", (1.0, 4.5))
    dist: list[float] = []

    def each(t: int, cmd: object, tel: object) -> None:
        c = world.walkers["child"]
        if t >= 190_000 and t % 1000 == 0 and c.here:
            dist.append(math.hypot(c.x - world.x, c.y - world.y))

    W.run_social(brain, world, 260, each=each)
    return brain, world, dist


def test_synthetic_a_sociable_hatchling_imprints_follows_and_settles(social: object) -> None:
    g = Genome.from_axes({"sociability": 0.95, "boldness": 0.4}, primary="snuggler", gd=7)
    brain, _world, dist = _synthetic_follow(g, seed=11)
    mom = brain.imprint.people.people.get(brain.imprint.mom) if brain.imprint.mom else None
    assert mom is not None and I.look_distance(mom.mean, LOOKS["child"]) < 1.0
    assert brain.counts.get("follow_mom", 0) >= 1
    assert brain.counts.get("settle_feet", 0) >= 1
    # after she walked across the room it caught up: the last 40 s mostly within 1.5 m of her
    assert sum(d <= 1.5 for d in dist[-40:]) >= 20


def test_synthetic_an_aloof_adult_does_not_imprint(social: object) -> None:
    g = Genome.from_axes({"sociability": 0.05, "boldness": 0.8}, primary="explorer", gd=220)
    brain, _world, _dist = _synthetic_follow(g, seed=11)
    assert brain.imprint.mom is None
    assert brain.counts.get("follow_mom", 0) == 0


def test_once_she_has_stopped_it_comes_on_to_her_feet() -> None:
    imp = _imprint("hatchling", 0.9, bold=0.6)
    _care_session(imp, "child", 60.0, random.Random(12))
    assert imp.mom is not None
    rng = random.Random(13)
    t = 100.0
    # she walks across in front of it, 1.5 m off: it follows at its comfort distance …
    for k in range(20):
        seen = _seen(60, "child", 0.6 - 0.06 * k, 1.5, rng)
        imp.observe(t, 0.1, (0.0, 0.0, 0.0), [seen], I.CareCues())
        t += 0.1
    walking = imp.features((0.0, 0.0, 0.0), 0.0)
    # … then she stands there: the leash shortens to its stop distance — it goes to her feet
    for _ in range(40):
        imp.observe(t, 0.1, (0.0, 0.0, 0.0), [_seen(60, "child", -0.6, 1.5, rng)], I.CareCues())
        t += 0.1
    stopped = imp.features((0.0, 0.0, 0.0), 0.0)
    bond = stopped["mom_bond"]
    assert walking["mom_moving"] > 0.0
    assert stopped["mom_moving"] == 0.0
    assert stopped["mom_far"] > 0.8 * bond > walking["mom_far"]
    assert stopped["mom_near_still"] == 0.0  # 1.5 m is not at her feet yet
    leash, _rise = imp.leash_m(bond, True)
    assert leash == pytest.approx(imp.stop_m() + I.LEASH_STOPPED_M)
    for _ in range(30):  # it has come to 1.0 m of her: it settles
        imp.observe(t, 0.1, (0.5, 0.0, 0.0), [_seen(60, "child", -0.9, 1.0, rng)], I.CareCues())
        t += 0.1
    near = imp.features((0.5, 0.0, 0.0), 0.0)
    assert near["mom_near_still"] > 0.5 and near["mom_far"] < 0.2


def test_a_mother_standing_still_does_not_walk_by_the_eyes_scatter() -> None:
    imp = _imprint("hatchling", 0.9, bold=0.3)
    _care_session(imp, "child", 60.0, random.Random(14))
    rng = random.Random(15)
    t = 100.0
    moving = 0
    for k in range(300):  # 30 s of looks at her, 1.6 m off, standing still: the eye's range
        # scatters 8 % and its bearing 0.02 rad (neuro_senses, bright light)
        rng_m = 1.6 * (1.0 + rng.gauss(0.0, 0.08))
        seen = _seen(60, "child", 0.1 + rng.gauss(0.0, 0.02), rng_m, rng)
        imp.observe(
            t, 0.1, (0.0, 0.0, 0.0), [seen] if k % 2 == 0 else [], I.CareCues(), fresh=k % 2 == 0
        )
        view = imp.mom_view((0.0, 0.0, 0.0))
        moving += view is not None and view.moving
        t += 0.1
    assert moving <= 15  # ≤ 5 % of the time (a flat 0.4 m made it "walk" every few seconds)


def test_synthetic_lost_it_goes_round_looking_and_finds_her(social: object) -> None:
    # she walks off behind it while it pecks; it looks where she stood, she is gone, it goes to
    # the spot and round in an arc (with looks) until it sees her, then goes to her.
    # A population test, not pinned seeds. Measured 2026-10-02 at 6ec7f91 over seeds 1-40:
    # found fast (<= 60 s) and close (<= 1.6 m) 38/40, went round looking 6/40, both together
    # 5/40. Going round looking is a rare event (she is usually seen again before the duck is
    # "lost"); the 9/29 baseline did it 10/40 and the floor-look corridor change (boundary.py,
    # 9/30) lowered it, which is not significant at n=40 and is noted for a later look.
    found = []
    for seed in range(1, 41):
        g = Genome.from_axes({"sociability": 0.9, "boldness": 0.3}, primary="snuggler", gd=7)
        brain = SB.SocialBrain(g, seed=seed)
        world = W.SocialSynth(seed=seed)
        world.care(10_000, "child", 100)
        st: dict[str, float] = {}

        def each(
            t: int,
            cmd: object,
            tel: object,
            world: W.SocialSynth = world,
            st: dict = st,
            brain: SB.SocialBrain = brain,
        ) -> None:
            c = world.walkers["child"]
            if t == 130_000:  # behind it, 1.6 m off, inside the room
                bx = world.x - 1.6 * math.cos(world.yaw)
                by = world.y - 1.6 * math.sin(world.yaw)
                c.goal = (min(max(bx, 0.5), W.ROOM - 0.5), min(max(by, 0.5), W.ROOM - 0.5))
            if t > 140_000 and "seen" not in st:
                view = brain.imprint.mom_view(brain._s_odom)
                if view is not None and view.seen:
                    st["seen"] = (t - 130_000) / 1000
            if t > 140_000 and c.here and t % 1000 == 0:
                st["d"] = math.hypot(c.x - world.x, c.y - world.y)

        W.run_social(brain, world, 230, each=each)
        scanned = any("look round for mom" in what for _t, what in brain.social_log)
        found.append((st.get("seen"), st.get("d"), scanned))
    fast_and_close = [
        1
        for seen, d, _s in found
        if seen is not None and seen <= 60.0 and d is not None and d <= 1.6
    ]
    assert len(fast_and_close) >= 34
    assert sum(1 for _seen, _d, scanned in found if scanned) >= 2
    assert (
        sum(
            1
            for seen, d, scanned in found
            if scanned and seen is not None and seen <= 60.0 and d is not None and d <= 1.6
        )
        >= 1
    )
