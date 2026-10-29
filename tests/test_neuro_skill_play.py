"""a:play — ball play, carrying a toy home and sunbathing wired into the neural brain
(duckmind.neuro.skill_play, skill_common, skill_ball). Pure: no physics."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from duckmind.neuro import brain as B
from duckmind.neuro import contract as C
from duckmind.neuro import genome as G
from duckmind.neuro import repertoire as P
from duckmind.neuro import select as S
from duckmind.neuro import skill_play as SP
from duckmind.neuro.skill_ball import PENDING_MS, BallPlay, BallStyle, ball_features
from duckmind.neuro.skill_common import OdomHistory, PlayThing, ThingBook

# ── the shared tables: in for the context, out after ─────────────────────────


def test_play_installed_adds_and_restores_every_table() -> None:
    before = (
        dict(C.BEHAVIOURS),
        tuple(S.FEATURES),
        dict(S.F),
        {k: dict(v) for k, v in S.W.items()},
        dict(S.SATIATE_S),
        dict(B.LOCO),
        dict(B.EXERTION),
        dict(P.BUILDERS),
        dict(G.ETHOGRAM_FIT),
        B.BEAK_BUSY,
        B.HEAD_ACTS,
        B.TURN_TOWARD_NEVER,
        P.REACTIVE,
        S.LOCOMOTES,
        G._gains,
    )
    assert "carry_toy" not in C.BEHAVIOURS
    with SP.play_installed():
        for act in SP.PLAY_ACTS:
            assert act in C.BEHAVIOURS and act in S.W and act in P.BUILDERS and act in B.LOCO
        for f in SP.PLAY_FEATURES:
            assert f in S.F and f in S.FEATURE_ZH
        assert S.FEATURES[: len(before[1])] == before[1]  # the old inputs keep their places
        assert "carry_toy" in B.BEAK_BUSY and "sunbathe" in B.HEAD_ACTS
        assert S.W["doze"]["in_sun"] == SP.PLAY_CROSS_W["doze"]["in_sun"]
        with SP.play_installed():  # nested: a no-op
            assert "sunbathe" in C.BEHAVIOURS
    after = (
        dict(C.BEHAVIOURS),
        tuple(S.FEATURES),
        dict(S.F),
        {k: dict(v) for k, v in S.W.items()},
        dict(S.SATIATE_S),
        dict(B.LOCO),
        dict(B.EXERTION),
        dict(P.BUILDERS),
        dict(G.ETHOGRAM_FIT),
        B.BEAK_BUSY,
        B.HEAD_ACTS,
        B.TURN_TOWARD_NEVER,
        P.REACTIVE,
        S.LOCOMOTES,
        G._gains,
    )
    assert after == before


def test_play_brain_needs_the_entries() -> None:
    g = G.preset("explorer")
    with pytest.raises(RuntimeError):
        SP.PlayBrain(g, 1)


def test_every_act_has_plain_words_and_a_program() -> None:
    for act, spec in SP.PLAY_BEHAVIOURS.items():
        assert spec.zh and spec.body_zh
        assert act in SP.PLAY_BUILDERS and act in SP.PLAY_LOCO and act in SP.PLAY_SATIATE_S
    for feature in SP.PLAY_FEATURES:
        assert SP.PLAY_FEATURE_ZH[feature]


# ── temperament: how likely, never whether ─────────────────────────────────


def _axes(**kw: float) -> dict[str, float]:
    base = {"boldness": 0.5, "activity": 0.5, "sociability": 0.5}
    base.update(kw)
    return base


def test_play_gain_leans_with_temperament() -> None:
    none = lambda name: 0.0  # noqa: E731
    bold = SP.play_gain("play_ball", _axes(boldness=0.9), "adult", none)
    shy = SP.play_gain("play_ball", _axes(boldness=0.1), "adult", none)
    assert bold > shy > 0.0
    young = SP.play_gain("play_ball", _axes(), "hatchling", none)
    assert young > SP.play_gain("play_ball", _axes(), "adult", none)
    social = SP.play_gain("carry_toy", _axes(sociability=0.9, activity=0.2), "adult", none)
    busy = SP.play_gain("carry_toy", _axes(sociability=0.1, activity=0.9), "adult", none)
    assert social > busy > 0.0
    assert SP.play_gain("carry_toy", _axes(), "hatchling", none) < SP.play_gain(
        "carry_toy", _axes(), "adult", none
    )  # a hatchling's beak is clumsy
    calm = SP.play_gain(
        "sunbathe", _axes(activity=0.1), "adult", lambda n: 1.0 if n == "dreamer" else 0.0
    )
    lively = SP.play_gain("sunbathe", _axes(activity=0.9), "adult", none)
    assert calm > lively > 0.0


def test_presets_differ_in_their_play_gains() -> None:
    with SP.play_installed():
        gains = {name: G.preset(name).gains for name in ("explorer", "snuggler", "dreamer")}
    assert gains["explorer"]["play_ball"] > gains["snuggler"]["play_ball"]
    assert gains["snuggler"]["carry_toy"] > gains["explorer"]["carry_toy"]
    assert gains["dreamer"]["sunbathe"] > gains["explorer"]["sunbathe"]


# ── the brain with the acts (no body: frames by hand) ────────────────────────


def _frame(t_ms: int, **kw: object) -> C.SensorFrame:
    body = C.BodySense(posture="stand")
    return C.SensorFrame(t_ms=t_ms, body=body, day_minute=600.0, **kw)  # type: ignore[arg-type]


def test_play_brain_steps_and_reports_its_play_state() -> None:
    with SP.play_installed():
        brain = SP.PlayBrain(G.preset("snuggler"), 3)
        for t in range(0, 4000, 50):
            cmd, tel = brain.step(t, _frame(t))
        assert "play" in tel.learned
        view = tel.learned["play"]
        assert view["version"] == SP.PLAY_VERSION
        assert view["home_mm"] is None  # it has not seen its dock
        feats = {f: brain._feat[S.F[f]] for f in SP.PLAY_FEATURES}
        assert feats["toy"] == 0.0 and feats["ball"] == 0.0 and feats["sunny"] == 0.0
        assert 0.0 < feats["nest_urge"] <= 1.0


def test_play_never_masks_an_unattributed_sound_or_contact_during_its_own_window() -> None:
    with SP.play_installed():
        brain = SP.PlayBrain(G.preset("snuggler"), 3)
        thud = C.SoundSense(level_db=67.0, onset=1.0, kind="bang")
        touch = C.TouchSense(site="back", pressure=0.9, source="model")
        frame = _frame(1000, sound=thud, touch=touch, events=("blocked",))
        brain._t_now = 1000
        brain.play.own_until = 1500  # it just let the toy go
        felt = brain._reafference(frame)
        assert felt.sound == thud
        assert felt.touch == touch
        assert "blocked" in felt.events
        brain._t_now = 3000
        assert brain._reafference(replace(frame, t_ms=3000)).sound.onset == 1.0


# ── the memory of things ──────────────────────────────────────────────────────


def _sight(
    bearing: float,
    rng: float,
    *,
    kind: str = "thing",
    motion: float = 0.0,
    track: int = 5,
    size: float = 0.25,
    axis: float | None = None,
):  # type: ignore[no-untyped-def]
    fields = dict(
        kind=kind,
        bearing=bearing,
        elevation=-0.5,
        range_m=rng,
        size=size,
        motion=motion,
        looming=0.0,
        track=track,
        source="model",
    )
    s = C.Sighting(**fields)  # type: ignore[arg-type]
    if axis is not None:
        s = _AxisSighting(**fields, axis=axis)  # type: ignore[arg-type]
    return s


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True, slots=True)
class _AxisSighting(C.Sighting):
    axis: float | None = None


def _see(
    book: ThingBook, hist: OdomHistory, t: int, odom, sightings, *, walking=False, dipped=False
):  # type: ignore[no-untyped-def]
    hist.add(t, odom)
    body = C.BodySense(
        posture="walk" if walking else "stand", odom=odom, vx=0.3 if walking else 0.0
    )
    cam = C.CameraSense(usable=True, sightings=tuple(sightings), age_ms=0)
    book.see(t, C.SensorFrame(t_ms=t, body=body, camera=cam), hist, dipped=dipped)


def test_walking_toward_a_still_thing_does_not_make_it_a_roller() -> None:
    book, hist = ThingBook(), OdomHistory()
    x = 0.0
    for k in range(20):  # walking at it; the eye calls it moving (it grows in the picture)
        t = 1000 + 200 * k
        x += 0.06
        _see(
            book,
            hist,
            t,
            (x, 0.0, 0.0),
            [_sight(0.0, 1.4 - x, kind="motion", motion=0.2)],
            walking=True,
        )
    assert all(th.rolled == 0 for th in book.things.values())


def test_the_book_ignores_passes_while_the_body_dips() -> None:
    book, hist = ThingBook(), OdomHistory()
    cam_body = C.BodySense(posture="stand")
    hist.add(0, (0.0, 0.0, 0.0))
    frame = C.SensorFrame(
        t_ms=0,
        body=cam_body,
        camera=C.CameraSense(
            usable=True,
            sightings=(
                C.Sighting(
                    kind="thing",
                    bearing=0.0,
                    elevation=-0.5,
                    range_m=0.3,
                    size=0.25,
                    motion=0.0,
                    looming=0.0,
                    track=1,
                    source="model",
                ),
            ),
            age_ms=0,
        ),
    )
    assert book.see(0, frame, hist, dipped=True) == []
    assert not book.things
    assert book.see(0, frame, hist) and len(book.things) == 1


def test_a_thing_just_out_of_its_beak_does_not_count_as_rolling() -> None:
    book = ThingBook()
    th = PlayThing(1, 0.3, 0.0, 0, 0, widest=0.08)
    th.carried_ms = 1000
    book.things[1] = th
    hist = OdomHistory()
    hist.add(1500, (0.0, 0.0, 0.0))
    frame = C.SensorFrame(
        t_ms=1500,
        body=C.BodySense(posture="stand"),
        camera=C.CameraSense(
            usable=True,
            sightings=(
                C.Sighting(
                    kind="motion",
                    bearing=0.0,
                    elevation=-0.5,
                    range_m=0.35,
                    size=0.25,
                    motion=0.4,
                    looming=0.0,
                    track=1,
                    source="model",
                ),
            ),
            age_ms=0,
        ),
    )
    book.see(1500, frame, hist)
    assert th.rolled == 0 and not th.moving(1500)


def test_ball_features_and_style() -> None:
    book = ThingBook()
    book.things[1] = PlayThing(1, 0.6, 0.0, 900, 0, rolled=1, fun=0.5)
    f = ball_features(book, (0.0, 0.0, 0.0), 1000)
    assert f["ball"] > 0.5 and f["ball_fun"] == 0.5 and f["ball_rolling"] == 0.0
    book.things[1].moving_ms = 950
    assert ball_features(book, (0.0, 0.0, 0.0), 1000)["ball_rolling"] > 0.5
    # a small untried round thing is worth a nudge; a long one less
    book2 = ThingBook()
    book2.things[2] = PlayThing(2, 0.5, 0.0, 900, 0, widest=0.07)
    round_ = ball_features(book2, (0.0, 0.0, 0.0), 1000)["ball"]
    book2.things[2].axis = 0.3
    long_ = ball_features(book2, (0.0, 0.0, 0.0), 1000)["ball"]
    assert round_ > long_ > 0.0
    bold = BallStyle.of({"boldness": 0.9, "activity": 0.8, "wit": 0.5}, 0.0)
    timid = BallStyle.of({"boldness": 0.1, "activity": 0.2, "wit": 0.5}, 0.0)
    assert bold.kick > timid.kick and timid.beak > bold.beak


def test_a_push_it_could_not_watch_end_is_settled_by_the_next_look() -> None:
    """A chest push can leave the ball nearer than the eye sees: the next look at that ball
    (whatever act runs then) says whether it went — ≥ 10 cm: it rolled; less: nothing."""
    book = ThingBook()
    ball = PlayThing(1, 0.10, 0.0, 1000, 0)
    book.things[1] = ball
    play = BallPlay(seed=3)
    play.pending = (1, 0.10, 0.0, 2000)
    assert not play.settle_pending(2500, book)  # not seen since the push: still open
    assert play.pending is not None
    ball.x, ball.seen_ms = 0.32, 3000  # seen again 22 cm on
    assert play.settle_pending(3000, book)
    assert play.rolls_seen == 1 and ball.rolled == 1 and ball.fun > 0.0 and play.pending is None
    # seen again where it lay: no roll, and the push is settled
    play.pending = (1, 0.32, 0.0, 4000)
    ball.x, ball.seen_ms = 0.36, 4500
    assert not play.settle_pending(4500, book) and play.pending is None and play.rolls_seen == 1
    # never seen again: forgotten after PENDING_MS
    play.pending = (1, 0.36, 0.0, 5000)
    assert not play.settle_pending(5000 + PENDING_MS + 1, book) and play.pending is None


def test_the_long_axis_is_averaged_over_still_looks() -> None:
    book, hist = ThingBook(), OdomHistory()
    axes = [0.40, 0.20, 0.35, 0.25, 0.30]
    for k, a in enumerate(axes):
        t = 1000 + 500 * k
        hist.add(t, (0.0, 0.0, 0.0))
        s = _AxisSighting(
            kind="thing",
            bearing=0.0,
            elevation=-0.5,
            range_m=0.5,
            size=0.2,
            motion=0.0,
            looming=0.0,
            track=7,
            source="model",
            axis=a,
        )
        cam = C.CameraSense(usable=True, sightings=(s,), age_ms=0)
        book.see(t, C.SensorFrame(t_ms=t, body=C.BodySense(posture="stand"), camera=cam), hist)
    th = next(iter(book.things.values()))
    assert th.axis is not None and abs(th.axis - 0.30) < 0.05
    assert th.axis_n == len(axes)
    assert math.isfinite(th.axis)


@pytest.mark.parametrize("reflex", ["orient", "lean_in"])
def test_play_reflex_coexist_keeps_only_leg_intent_and_target(reflex: str) -> None:
    target = C.Target(kind="ball", bearing=0.2, range_m=0.4, track=7)
    cmd = C.MotorCommand(
        t_ms=100,
        behaviour="play_ball",
        bout=2,
        variant=4,
        vx=0.35,
        vyaw=0.1,
        reflex=reflex,
        vocal="chirp",
        moment="head_tilt",
        moment_seq=8,
    )
    out = SP.PlayOut(
        drive=True,
        vx=0.4,
        vyaw=-0.2,
        fine=True,
        target=target,
        head=(0.0, 0.2, 0.1, 0.0),
        gaze=C.Gaze(),
        mouth=0.8,
        skill="kick_left",
        skill_seq=3,
        posture="sit",
        vocal="wheee",
        quiet=True,
    )
    safe = SP.PlayBrain._reflex_safe_out(out, cmd, "play_ball")
    assert (safe.drive, safe.vx, safe.vyaw, safe.fine, safe.target) == (
        True,
        0.4,
        -0.2,
        True,
        target,
    )
    assert (safe.head, safe.gaze, safe.mouth, safe.skill, safe.skill_seq) == (
        None,
        None,
        None,
        None,
        0,
    )
    assert (safe.posture, safe.vocal, safe.quiet) == (None, None, False)

    merged = SP.PlayBrain._merge(object.__new__(SP.PlayBrain), cmd, safe, "play_ball")
    assert (merged.vx, merged.vyaw, merged.target) == (0.4, -0.2, target)
    assert (merged.reflex, merged.vocal, merged.moment, merged.moment_seq) == (
        reflex,
        "chirp",
        "head_tilt",
        8,
    )


@pytest.mark.parametrize("reflex", ["veer", "stabilise", "startle", "withdraw"])
def test_play_reflex_other_than_explicit_head_only_pair_keeps_ownership(reflex: str) -> None:
    cmd = C.MotorCommand(t_ms=100, behaviour="play_ball", bout=2, variant=4, vx=0.0, reflex=reflex)
    out = SP.PlayOut(drive=True, vx=0.4, vyaw=0.2, fine=True, skill="kick_left", skill_seq=9)
    assert SP.PlayBrain._reflex_safe_out(out, cmd, "play_ball") == SP.IDLE
    merged = SP.PlayBrain._merge(object.__new__(SP.PlayBrain), cmd, out, "play_ball")
    assert merged == cmd
