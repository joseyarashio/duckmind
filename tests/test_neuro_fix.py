"""neuro:fix (2026-09-28) — the review's findings, each held by a test: things known again by
where they are, a beak that closes, moments inside an act, a head that holds a fixation, bouts
used up by their need, the family as a safety signal and a draw, curiosity's back arc only
beside a thing, a cornered duck that stops instead of pressing, caution where it has not seen
the floor, a body without skin, and calls chosen by what is going on."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from duckmind.neuro import boundary as B
from duckmind.neuro import contract as C
from duckmind.neuro import motor as M
from duckmind.neuro import reflex as R
from duckmind.neuro import repertoire as P
from duckmind.neuro.bodylink import shape_twist
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import Genome, preset
from duckmind.neuro.learn import Learner
from duckmind.neuro.synth import SynthWorld


def _genome(boldness: float = 0.5, sociability: float = 0.5, gd: int = 120) -> Genome:
    return Genome.from_axes(
        {"boldness": boldness, "sociability": sociability}, primary="songbird", gd=gd
    )


# ── things are known again by where they are ─────────────────────────────────


def test_a_thing_is_known_again_by_where_it_is_not_by_the_eyes_track_id() -> None:
    """The eye re-issues a track id whenever a thing leaves the picture for a moment (review:
    the same toy was "never seen" 36 times in 12 min): the brain knows it again by place."""
    learn = Learner(_genome().p)
    odom = (0.0, 0.0, 0.0)
    key1, fresh = learn.see_thing(5, "thing", odom, 1.0, 0.0, 0.0, 0.5, 0.0)
    for k in range(200):  # it has a good look (20 s)
        learn.see_thing(5, "thing", odom, 1.0, 0.0, 0.1 * k, 0.1, 0.0)
    key2, again = learn.see_thing(9, "thing", odom, 1.05, 0.03, 30.0, 0.1, 0.0)  # a new track
    assert key2 == key1 and len(learn.things) == 1
    assert again < 0.6 * fresh  # not "never seen" any more
    assert learn.reappeared > 0.9  # "there it is again": a double take
    key3, other = learn.see_thing(12, "thing", odom, 3.0, 2.0, 31.0, 0.1, 0.0)  # elsewhere
    assert key3 != key1 and other > 0.9


# ── the beak ──────────────────────────────────────────────────────────────────


def test_a_beak_left_open_by_a_cut_short_peck_closes_at_once() -> None:
    style = P.style_from({"activity": 0.5}, stage="juvenile", side=1)
    frame = C.SensorFrame(t_ms=0, body=C.BodySense(posture="stand"))
    cmd = C.MotorCommand(t_ms=0, behaviour="peck", bout=1, variant=3)
    state, out = P.step(None, cmd, style, frame, 0)
    t = 0
    while out.pose.mouth < 0.2 and t < 8000:  # until the beak is open mid-peck
        t += 20
        state, out = P.step(state, replace(cmd, t_ms=t), style, frame, t)
    assert out.pose.mouth >= 0.2
    walk = C.MotorCommand(t_ms=t, behaviour="wander", bout=2, variant=4)
    for dt in range(20, 320, 20):
        state, out = P.step(state, replace(walk, t_ms=t + dt), style, frame, t + dt)
    assert out.pose.mouth < 0.05  # shut within 0.3 s


# ── moments play inside an act ────────────────────────────────────────────────


def test_moments_are_events_inside_the_act_not_acts_of_their_own() -> None:
    brain = NeuroBrain(preset("explorer"), seed=2)
    world = SynthWorld(seed=2)
    seqs = set()
    labels = set()
    t = 0
    while t < 600_000:
        cmd, tel = brain.step(t, world.frame(t))
        world.apply(cmd, 0.1)
        labels.add(cmd.behaviour)
        if cmd.moment:
            seqs.add((cmd.moment_seq, cmd.moment))
        t += 100
    assert len(seqs) >= 10  # it glances, tilts, looks down… many times
    assert {m for _, m in seqs} <= C.MOMENTS
    assert not labels & C.MOMENTS  # … and they never replace what it is doing


# ── the head holds a fixation ─────────────────────────────────────────────────


def test_between_gaze_shifts_the_head_holds_still() -> None:
    brain = NeuroBrain(_genome(), seed=4)
    world = SynthWorld(seed=4, family=False)
    world.toys = []
    brain.sel.current = brain.sel.index("rest_sit")
    brain.behaviour = "rest_sit"
    bearings = []
    t = 0
    while t < 60_000:
        cmd, _ = brain.step(t, world.frame(t))
        world.apply(cmd, 0.05)
        bearings.append(round(cmd.gaze.bearing, 6))
        t += 50
    changes = sum(1 for a, b in zip(bearings, bearings[1:], strict=False) if a != b)
    assert changes < 0.2 * len(bearings)  # shifts, then holds (no drifting fixation)


# ── bouts are used up by their need ───────────────────────────────────────────


def test_a_preen_lasts_while_the_itch_lasts() -> None:
    """The preen bout is satisfied by the itch it scratches (drives.groom): a big itch gives a
    long preen, a small one a short one — the same duck, the same clock."""

    def preen_s(groom: float) -> float:
        brain = NeuroBrain(_genome(boldness=0.5), seed=7)
        world = SynthWorld(seed=7, family=False)
        world.toys = []
        brain.drives.groom = groom
        brain.drives.restless = 0.0
        brain.drives.curiosity = 0.0
        brain.sel.current = brain.sel.index("preen")
        brain.behaviour = "preen"
        brain._bout = {"need0": groom, "t_s": 0.0}
        t = 0
        while t < 120_000:
            cmd, _ = brain.step(t, world.frame(t))
            world.apply(cmd, 0.1)
            if cmd.behaviour != "preen":
                return t / 1000.0
            t += 100
        return 120.0

    assert preen_s(1.0) > 1.5 * preen_s(0.3)


# ── the family: a safety signal, and a draw ───────────────────────────────────


def test_its_own_name_never_startles_even_a_shy_duck() -> None:
    params = R.params_from_genome(_genome(boldness=0.0, sociability=0.9, gd=20))
    loud_name = C.SensorFrame(t_ms=0, sound=C.SoundSense(level_db=82.0, onset=1.0, kind="name"))
    for seed in range(20):
        out = R.step(params, R.initial(seed), loud_name, hab={}, sens={})[1]
        assert out.events and all(e.reflex == "orient" for e in out.events)


def test_a_social_duck_comes_to_the_family_it_sees_when_called() -> None:
    came = 0
    for seed in range(1, 5):
        world = SynthWorld(seed=seed, family=True, day_minute0=600)
        world._plan_t = world._next_talk = world._next_pet = 10**12
        world.x, world.y, world.yaw = 2.0, 3.0, 0.0
        world._home = True
        world._person = [3.6, 3.0, 0.0, 0.0]  # 1.6 m straight ahead, in view
        world._goal = (3.6, 3.0)
        brain = NeuroBrain(_genome(boldness=0.5, sociability=0.9), seed=seed)
        motor = M.MotorSystem(M.profile_from_genome(brain.genome), seed)
        t = 0
        while t < 45_000:
            if t == 20_000:
                world.call_name(t)
            frame = world.frame(t)
            cmd, _ = brain.step(t, frame)
            out = motor.step(cmd, frame)
            world.apply(replace(cmd, vx=out.vx, vyaw=out.vyaw), 0.05, out.head_goal[2])
            t += 50
        came += math.hypot(world._person[0] - world.x, world._person[1] - world.y) < 1.1
    assert came >= 3


# ── curiosity's back arc: only beside a thing, only an arc ────────────────────


def test_a_reverse_is_only_a_docking_reverse_or_curiositys_back_arc() -> None:
    assert C.legal_walk(-0.4, 0.8, back_arc=True) == (-0.4, C.BACK_ARC_VYAW)
    assert C.legal_walk(-0.4, 0.0, back_arc=True) == (0.0, 0.0)  # never straight back
    assert C.legal_walk(-0.4, 0.8) == (0.0, 0.0)
    vx, vyaw, notes = shape_twist(-0.4, -1.0, back_arc=True)
    assert (vx, vyaw) == (-0.4, -1.0) and "back_arc" in notes
    assert shape_twist(-0.4, -0.2, back_arc=True)[0] == 0.0
    assert shape_twist(-0.4, -1.0)[0] == 0.0


def test_done_with_a_thing_right_in_front_it_backs_off_in_an_arc() -> None:
    brain = NeuroBrain(preset("explorer"), seed=3)
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
    brain._thing_last = (0.1, 0.25, 0.0)
    brain._maybe_back_off(1000, 0.5, (0.0, 0.0, 0.0), True)
    assert brain._back is not None
    frame = C.SensorFrame(t_ms=1100, body=C.BodySense(posture="stand"))
    cmd, _ = brain.step(1100, frame)
    assert cmd.back_arc and cmd.vx < 0.0 and abs(cmd.vyaw) == pytest.approx(C.BACK_ARC_VYAW)
    far = NeuroBrain(preset("explorer"), seed=3)
    far._thing_last = (0.1, 0.9, 0.0)  # not right in front: no reversing
    far._maybe_back_off(1000, 0.5, (0.0, 0.0, 0.0), True)
    assert far._back is None


# ── walls: cornered it stops; in the unseen it slows and looks ahead ───────────


def _walls_all_round(radius: float) -> B.Boundary:
    b = B.Boundary()
    for k in range(48):
        a = 2 * math.pi * k / 48
        b._add((0.0, 0.0, 0.0), a, radius, "wall", 1.0)
    return b


def test_cornered_it_stops_instead_of_pressing_into_the_walls() -> None:
    b = _walls_all_round(0.3)
    steer = b.steer(0, (0.0, 0.0, 0.0))
    assert steer.cornered
    open_room = _walls_all_round(1.2)
    assert not open_room.steer(0, (0.0, 0.0, 0.0)).cornered


def test_where_it_has_not_seen_the_floor_it_slows_and_looks_ahead() -> None:
    unseen = B.Boundary().steer(0, (0.0, 0.0, 0.0))
    assert unseen.look_ahead and unseen.cap == pytest.approx(B.APPROACH_CAP)
    seen = B.Boundary()
    line = tuple(
        C.FreeSector(bearing=b, free_m=1.5, end="unknown") for b in (0.3, 0.15, 0.0, -0.15, -0.3)
    )
    seen.update(
        0,
        C.SensorFrame(
            t_ms=0,
            body=C.BodySense(posture="stand"),
            camera=C.CameraSense(usable=True, floor=line, floor_t_ms=0),
        ),
    )
    clear = seen.steer(200, (0.0, 0.0, 0.0))
    assert not clear.look_ahead and clear.cap is None


def test_a_slot_beside_the_furniture_is_no_way_out() -> None:
    """A narrow opening (a slot between the sofa and the wall) looks clear straight along it but
    the body cannot turn in it: it does not steer into it."""
    b = B.Boundary()
    for along in (0.3, 0.45, 0.6, 0.75, 0.9):
        for across in (-0.12, 0.12):
            b._add(
                (0.0, 0.0, 0.0), math.atan2(across, along), math.hypot(along, across), "wall", 1.0
            )
    steer = b.steer(0, (0.0, 0.0, 0.0))
    assert abs(steer.best) > 0.25  # it bends away from the slot, not into it


# ── the body has no skin ──────────────────────────────────────────────────────


def test_a_gentle_touch_on_the_body_is_felt_late_and_unplaced() -> None:
    from duckmind.world import neuro_senses as ns

    body_hits, head_hits = [], []
    for seed in range(30):
        model = ns.TouchModel(seed=seed)
        model.update(
            [
                {
                    "key": "hand",
                    "t0": 0.0,
                    "t1": 2.0,
                    "site": "back",
                    "active": True,
                    "depth_mm": 1.0,
                    "now_depth_mm": 1.0,
                    "closing": 0.0,
                    "slide": 0.1,
                    "now_slide": 0.1,
                }
            ]
        )
        felt = next((t / 1000 for t in range(0, 2000, 10) if model.step(t / 1000).site), None)
        if felt is not None:
            body_hits.append((felt, model.step(felt).site))
        head = ns.TouchModel(seed=seed)
        head.update(
            [
                {
                    "key": "hand",
                    "t0": 0.0,
                    "t1": 2.0,
                    "site": "head",
                    "active": True,
                    "depth_mm": 1.0,
                    "now_depth_mm": 1.0,
                    "closing": 0.0,
                    "slide": 0.1,
                    "now_slide": 0.1,
                }
            ]
        )
        felt = next((t / 1000 for t in range(0, 2000, 10) if head.step(t / 1000).site), None)
        if felt is not None:
            head_hits.append((felt, head.step(felt).site))
    assert body_hits and all(site == "body" and at >= 0.25 for at, site in body_hits)
    assert head_hits and all(site in ("head", "beak") and at <= 0.09 for at, site in head_hits)


# ── calls fit what is going on ────────────────────────────────────────────────


def test_an_arrival_draws_greetings_not_only_its_favourite_call() -> None:
    brain = NeuroBrain(preset("explorer"), seed=5)  # its favourite call is "inquire"
    tags = ["inquire", "greet", "chirp"]
    calm = [brain._tag_by_context(tags, "wander", brain.drives) for _ in range(200)]
    brain._traces["person_new"] = 1.0
    arrived = [brain._tag_by_context(tags, "wander", brain.drives) for _ in range(200)]
    assert arrived.count("greet") > 1.8 * calm.count("greet")
    assert len(set(calm)) == 3  # sampled, never the same argmax every time
