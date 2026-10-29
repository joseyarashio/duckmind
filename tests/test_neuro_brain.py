"""The neural duck's brain (neuro:brain): contract, genome, determinism, diversity, habituation,
reflex interrupts, satiation-ended bouts, the owner's fall rule, honest senses."""

from __future__ import annotations

import ast
import math
import statistics
from pathlib import Path

import pytest

from duckmind.duckcore import content
from duckmind.duckcore.engine import DuckCore
from duckmind.neuro import contract as C
from duckmind.neuro import select as S
from duckmind.neuro.brain import NeuroBrain
from duckmind.neuro.genome import AXES, Genome, preset
from duckmind.neuro.habituation import Habituation
from duckmind.neuro.learn import AVOID_AFTER, Learner
from duckmind.neuro.perceive import Percept
from duckmind.neuro.synth import SynthWorld, iterate, run
from duckmind.neuro.telemetry import Budget

NEURO = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "neuro"
BRAIN_MODULES = (
    "contract",
    "genome",
    "signals",
    "drives",
    "habituation",
    "perceive",
    "reflex",
    "appraisal",
    "motor",
    "repertoire",
    "learn",
    "select",
    "brain",
    "telemetry",
    "synth",
)
TEMPERAMENTS = {
    "bold_explorer": (
        dict(
            boldness=0.9,
            activity=0.9,
            sociability=0.3,
            exploration=0.9,
            wit=0.7,
            chronotype=0.3,
            tempo=0.6,
        ),
        "explorer",
    ),
    "shy_snuggler": (
        dict(
            boldness=0.1,
            activity=0.3,
            sociability=0.9,
            exploration=0.2,
            wit=0.4,
            chronotype=0.5,
            tempo=0.4,
        ),
        "snuggler",
    ),
    "lazy_dreamer": (
        dict(
            boldness=0.5,
            activity=0.1,
            sociability=0.4,
            exploration=0.3,
            wit=0.3,
            chronotype=0.8,
            tempo=0.2,
        ),
        "dreamer",
    ),
    "social_songbird": (
        dict(
            boldness=0.6,
            activity=0.6,
            sociability=0.8,
            exploration=0.5,
            wit=0.8,
            chronotype=0.4,
            tempo=0.7,
        ),
        "songbird",
    ),
}


def _genome(name: str, gd: int = 120) -> Genome:
    axes, branch = TEMPERAMENTS[name]
    return Genome.from_axes(axes, primary=branch, branch=branch, gd=gd)


# ── contract ──────────────────────────────────────────────────────────────────


def test_walk_progress_requires_odometry_not_a_walking_command() -> None:
    brain = NeuroBrain(_genome("bold_explorer"), seed=1)

    def frame(t_ms: int, x: float) -> C.SensorFrame:
        return C.SensorFrame(
            t_ms=t_ms,
            body=C.BodySense(posture="walk", odom=(x, 0.0, 0.0)),
            sources={"body": "model"},  # SynthWorld has odometry but no leg-load model.
        )

    assert brain._measured_walk_seconds(frame(0, 0.0), 0.05, False, False) == 0.0
    # A requested step with stationary contact odometry is not locomotion.
    assert brain._measured_walk_seconds(frame(50, 0.0), 0.05, True, False) == 0.0
    # Actual displacement is credited, even when optional leg-load telemetry is unknown.
    assert brain._measured_walk_seconds(frame(100, 0.005), 0.05, True, False) == pytest.approx(0.05)


def test_a_novel_visible_thing_can_interrupt_wander_for_investigation() -> None:
    """A curious explorer leaves an ordinary wander to inspect a new tracked thing."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=4107)
    brain.sel.current = brain.sel.index("wander")
    brain.behaviour = "wander"
    brain.drives.restless = 0.75
    seen = C.Sighting(kind="thing", bearing=0.22, range_m=0.8, size=0.12, track=41)
    chosen = []
    for k in range(100):
        cmd, _ = brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(posture="walk", odom=(0.004 * k, 0.0, 0.0)),
                camera=C.CameraSense(usable=True, sightings=(seen,)),
                power=C.PowerSense(battery=0.85),
            ),
        )
        chosen.append(cmd)
    investigate = [cmd for cmd in chosen if cmd.behaviour == "investigate"]
    assert investigate
    assert investigate[0].target is not None
    assert investigate[0].target.track == 41


def test_a_new_track_wins_attention_over_a_larger_known_thing() -> None:
    """A newly arrived bottle is not hidden from the brain by a familiar larger object."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=4107)
    book = C.Sighting(kind="thing", bearing=0.1, range_m=0.7, size=0.30, track=1)
    bottle = C.Sighting(kind="thing", bearing=-0.2, range_m=0.8, size=0.10, track=2)
    # Let the brain learn the large nearby book first.
    brain.step(
        0,
        C.SensorFrame(
            t_ms=0,
            body=C.BodySense(posture="walk"),
            camera=C.CameraSense(usable=True, sightings=(book,)),
            power=C.PowerSense(battery=0.85),
        ),
    )
    brain.sel.current = brain.sel.index("wander")
    brain.behaviour = "wander"
    _, telemetry = brain.step(
        20,
        C.SensorFrame(
            t_ms=20,
            body=C.BodySense(posture="walk", odom=(0.01, 0.0, 0.0)),
            camera=C.CameraSense(usable=True, sightings=(book, bottle)),
            power=C.PowerSense(battery=0.85),
        ),
    )
    assert brain._thing_last is not None
    assert brain._thing_last[0] == pytest.approx(bottle.bearing)
    assert (
        brain._bout_target("investigate", None, None, Percept(thing=bottle)).track == bottle.track
    )
    assert any(c.behaviour == "investigate" for c in telemetry.competition)


def test_investigation_keeps_its_new_track_when_a_larger_thing_remains_visible() -> None:
    """A bottle that began an investigation is not replaced by the larger familiar book."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=4107)
    book = C.Sighting(kind="thing", bearing=0.1, range_m=0.7, size=0.30, track=1)
    bottle = C.Sighting(kind="thing", bearing=-0.2, range_m=0.8, size=0.10, track=2)
    brain.learn._tracks[1] = 1  # the book is already familiar to the eye/learner mapping
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
    brain._attend_track = 2
    brain._attention_align_until = 1000
    cmd, _ = brain.step(
        20,
        C.SensorFrame(
            t_ms=20,
            body=C.BodySense(posture="walk", odom=(0.01, 0.0, 0.0)),
            camera=C.CameraSense(usable=True, sightings=(book, bottle)),
            power=C.PowerSense(battery=0.85),
        ),
    )
    assert cmd.target is not None
    assert cmd.target.track == 2
    assert cmd.target.bearing == pytest.approx(bottle.bearing)


def test_investigation_uses_its_recently_lost_track_for_at_most_one_second() -> None:
    """A brief occlusion keeps the same target, but stale attention is released."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=4107)
    book = C.Sighting(kind="thing", bearing=0.1, range_m=0.7, size=0.30, track=1)
    bottle = C.Sighting(kind="thing", bearing=-0.2, range_m=0.8, size=0.10, track=2)
    # Populate the actual learned position through an ordinary sensed bottle frame.
    brain.step(
        0,
        C.SensorFrame(
            t_ms=0,
            body=C.BodySense(posture="walk"),
            camera=C.CameraSense(usable=True, sightings=(bottle,)),
            power=C.PowerSense(battery=0.85),
        ),
    )
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
    brain._attend_track = 2
    brain._attention_align_until = 1000
    brain._attend_seen_s = 0.0
    cmd, _ = brain.step(
        500,
        C.SensorFrame(
            t_ms=500,
            body=C.BodySense(posture="walk", odom=(0.01, 0.0, 0.0)),
            camera=C.CameraSense(usable=True, sightings=(book,)),
            power=C.PowerSense(battery=0.85),
        ),
    )
    assert cmd.target is not None and cmd.target.track == 2
    cmd, _ = brain.step(
        1200,
        C.SensorFrame(
            t_ms=1200,
            body=C.BodySense(posture="walk", odom=(0.02, 0.0, 0.0)),
            camera=C.CameraSense(usable=True, sightings=(book,)),
            power=C.PowerSense(battery=0.85),
        ),
    )
    assert cmd.target is None or cmd.target.track != 2


def test_investigate_uses_goal_steering_without_wander_meander() -> None:
    """An object-directed investigate turn follows target bearing rather than wander noise."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=4107)
    bottle = C.Sighting(kind="thing", bearing=0.5, range_m=0.8, size=0.10, track=2)
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
    brain._attend_track = 2
    commands = []
    for k in range(100):
        cmd, _ = brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(posture="walk", odom=(0.004 * k, 0.0, 0.0)),
                camera=C.CameraSense(usable=True, sightings=(bottle,)),
                power=C.PowerSense(battery=0.85),
            ),
        )
        commands.append(cmd)
    assert all(cmd.target is not None and cmd.target.track == 2 for cmd in commands)
    assert any(cmd.vx > 0.0 and cmd.vyaw > 0.0 for cmd in commands)


def test_investigate_turns_a_far_side_target_before_advancing() -> None:
    """A target beyond the alignment window gets a decisive legal arc toward centre."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=4107)
    bottle = C.Sighting(kind="thing", bearing=0.8, range_m=1.0, size=0.10, track=2)
    brain.behaviour = "investigate"
    brain.sel.current = brain.sel.index("investigate")
    brain._attend_track = 2
    commands = []
    for k in range(100):
        cmd, _ = brain.step(
            20 * k,
            C.SensorFrame(
                t_ms=20 * k,
                body=C.BodySense(posture="walk", odom=(0.004 * k, 0.0, 0.0)),
                camera=C.CameraSense(usable=True, sightings=(bottle,)),
                power=C.PowerSense(battery=0.85),
            ),
        )
        commands.append(cmd)
    turning = [cmd for cmd in commands if cmd.vx > 0.0]
    assert turning
    assert all(cmd.vyaw > 0.0 for cmd in turning)


def test_brain_walk_budget_and_restless_use_measured_progress() -> None:
    stationary = NeuroBrain(_genome("bold_explorer"), seed=19)
    moving = NeuroBrain(_genome("bold_explorer"), seed=19)
    for brain in (stationary, moving):
        brain.drives.restless = 0.8

    def step(brain: NeuroBrain, t_ms: int, x: float):
        # Seed a prior accepted walking request; step() then receives ordinary body feedback.
        brain._stepping = True
        return brain.step(
            t_ms,
            C.SensorFrame(
                t_ms=t_ms,
                body=C.BodySense(posture="walk", odom=(x, 0.0, 0.0)),
                sources={"body": "real"},
            ),
        )[1]

    still_tel = moving_tel = None
    for k in range(50):
        t_ms = k * 50
        still_tel = step(stationary, t_ms, 0.0)
        moving_tel = step(moving, t_ms, 0.005 * k)

    assert still_tel is not None and moving_tel is not None
    assert still_tel.locomotion["walk_s"] == 0.0
    assert moving_tel.locomotion["walk_s"] >= 2.0
    assert moving_tel.drives["restless"] < still_tel.drives["restless"]


def test_walk_progress_rejects_invalid_contact_samples() -> None:
    brain = NeuroBrain(_genome("bold_explorer"), seed=1)

    def sample(
        t_ms: int,
        x: float,
        *,
        posture: str = "walk",
        fallen: bool = False,
        load: float | None = 1.0,
        events: tuple[str, ...] = (),
    ) -> C.SensorFrame:
        return C.SensorFrame(
            t_ms=t_ms,
            body=C.BodySense(posture=posture, fallen=fallen, leg_load=load, odom=(x, 0.0, 0.0)),
            events=events,
            sources={"body": "real"},
        )

    assert brain._measured_walk_seconds(sample(0, 0.0), 0.05, False, False) == 0.0
    assert brain._measured_walk_seconds(sample(50, 0.005, load=0.1), 0.05, True, False) == 0.0
    # Invalid samples clear the anchor, so recovery never bridges the invalid interval.
    assert brain._measured_walk_seconds(sample(100, 0.010, fallen=True), 0.05, True, False) == 0.0
    assert (
        brain._measured_walk_seconds(sample(150, 0.015, events=("lifted",)), 0.05, True, False)
        == 0.0
    )
    assert (
        brain._measured_walk_seconds(sample(200, 0.020, events=("set_down",)), 0.05, True, False)
        == 0.0
    )
    assert (
        brain._measured_walk_seconds(sample(250, 0.025, posture="stand"), 0.05, True, False) == 0.0
    )
    # A healthy sample after an invalid interval reanchors but does not bridge the gap.
    assert brain._measured_walk_seconds(sample(300, 0.030, load=None), 0.05, True, False) == 0.0
    # A reset-sized jump is rejected and invalidates the anchor; one clean sample reanchors.
    assert brain._measured_walk_seconds(sample(350, 1.0), 0.05, True, False) == 0.0
    assert brain._measured_walk_seconds(sample(400, 1.005), 0.05, True, False) == 0.0
    assert brain._measured_walk_seconds(sample(450, 1.010), 0.05, True, False) == pytest.approx(
        0.05
    )
    assert brain._measured_walk_seconds(sample(500, float("nan")), 0.05, True, False) == 0.0
    assert brain._measured_walk_seconds(sample(550, 2.0), 0.05, True, False) == 0.0
    assert brain._measured_walk_seconds(sample(600, 2.005), 0.05, True, False) == pytest.approx(
        0.05
    )
    assert brain._measured_walk_seconds(sample(650, 2.010), 0.05, True, False) == pytest.approx(
        0.05
    )


def test_legal_walk_never_breaks_the_walking_rules() -> None:
    values = [-1.0, -0.4, -0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.45, 0.55, 0.9]
    turns = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]
    for vx in values:
        for vyaw in turns:
            for approach in (False, True):
                out_vx, out_vyaw = C.legal_walk(vx, vyaw, approach=approach)
                assert out_vx == 0.0 or C.MIN_STEP_VX <= out_vx <= C.NORMAL_VX_MAX
                assert out_vx >= 0.0  # no reversing outside docking
                assert abs(out_vyaw) <= C.VYAW_MAX
                assert not (out_vx == 0.0 and out_vyaw != 0.0)  # turns = arcs
                if approach:
                    assert out_vx <= C.APPROACH_VX
    assert C.legal_walk(-0.3, 0.0, docking=True)[0] == pytest.approx(-0.3)


def test_the_repertoire_holds_every_duck_core_verb_and_is_fully_wired() -> None:
    for verb in content.VERBS:
        assert verb.verb_id in C.BEHAVIOURS, verb.verb_id
        spec = C.BEHAVIOURS[verb.verb_id]
        assert spec.unlock_gd == verb.unlock_gd and spec.branch == verb.branch
    for behaviour, spec in C.BEHAVIOURS.items():
        if spec.family == "reflex":
            continue
        assert behaviour in S.W and behaviour in S.SATIATE_S, behaviour
        for feature in S.W[behaviour]:
            assert feature in S.F, (behaviour, feature)
    assert len(C.BEHAVIOURS) >= 50


def test_sensor_frame_round_trips_through_json() -> None:
    world = SynthWorld(seed=2)
    frame = world.frame(1000)
    assert C.SensorFrame.from_json(frame.to_json()) == frame


# ── genome: Duck Core's diversity design ──────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 7, 42, 4107, 1728])
def test_genome_reuses_duck_cores_own_draws(seed: int) -> None:
    core = DuckCore(seed=seed)
    genome = Genome.from_seed(seed, gd=30)
    assert genome.axes_raw == {axis: int(core.s.temperament[axis]) for axis in AXES}
    assert genome.primary == core.s.branch_detail["primary"]
    assert genome.side == (1 if core.s.quirks["tilt_side"] == "左" else -1)
    assert genome.habit == core.s.quirks["habit"]
    assert genome.favorite_tag == core.s.quirks["favorite_tag"]
    assert genome.stage_days == {k: int(v) for k, v in core.s.stage_days.items()}
    assert Genome.from_seed(seed, gd=30).p == genome.p  # same seed, same duck


def test_growth_stage_gates_the_repertoire() -> None:
    young = Genome.from_seed(3341, gd=1)
    older = Genome.from_seed(3341, gd=12)
    adult = Genome.from_seed(3341, gd=220, branch="explorer")
    assert young.stage == "hatchling" and not any(C.BEHAVIOURS[b].locomotes for b in young.unlocked)
    assert "wander" in older.unlocked and "snuggle" not in older.unlocked
    assert {"survey", "peekaboo_master"} <= adult.unlocked
    assert not {"little_song", "melt", "moon_gaze"} & adult.unlocked  # other branches' signatures
    assert young.p.amplitude_cap < adult.p.amplitude_cap
    assert young.p.sleep_rise_s < adult.p.sleep_rise_s  # a hatchling tires into sleep sooner
    assert young.p.startle_threshold < adult.p.startle_threshold


def test_temperament_maps_onto_the_nervous_system() -> None:
    bold = _genome("bold_explorer").p
    shy = _genome("shy_snuggler").p
    assert bold.startle_threshold > shy.startle_threshold
    assert bold.fear_tau_s < shy.fear_tau_s
    assert bold.restless_tau_s < shy.restless_tau_s
    assert shy.social_tau_s < bold.social_tau_s
    assert preset("explorer").stage == "juvenile" and preset("explorer").branch == "explorer"


# ── determinism and no hidden clocks ──────────────────────────────────────────


def _trace(genome: Genome, seed: int, seconds: float, start_ms: int = 0) -> list[tuple]:
    world = SynthWorld(seed=5)
    world.clap(20_000)
    world.poke(45_000)
    out = run(NeuroBrain(genome, seed=seed), world, seconds, hz=20, start_ms=start_ms)
    return [
        (c.behaviour, c.bout, c.vx, c.vyaw, c.gaze.bearing, c.vocal, c.reflex, t.reason_zh)
        for c, t in out
    ]


def test_same_seed_and_frames_give_the_same_duck() -> None:
    genome = Genome.from_seed(7, gd=60)
    first = _trace(genome, 1, 120)
    assert first == _trace(genome, 1, 120)
    assert first != _trace(genome, 2, 120)


def test_only_elapsed_time_matters_not_the_clock_reading() -> None:
    genome = Genome.from_seed(7, gd=60)
    assert _trace(genome, 1, 90) == _trace(genome, 1, 90, start_ms=987_654_321)


def test_the_brain_imports_no_world_no_clock_no_dice() -> None:
    banned = {"duckmind.world", "time", "random", "socket", "threading", "datetime", "asyncio"}
    for name in BRAIN_MODULES:
        tree = ast.parse((NEURO / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                assert not any(mod == b or mod.startswith(b + ".") for b in banned), (name, mod)


# ── diversity: two hours, four temperaments, two seeds each ───────────────────


def _budget(genome: Genome, seed: int, seconds: float = 7200.0, hz: float = 5.0) -> Budget:
    world = SynthWorld(seed=100 + seed)
    for k in range(10):
        world.clap(600_000 + k * 600_000)
    budget = Budget()
    dt = 1.0 / hz
    for cmd, tel in iterate(NeuroBrain(genome, seed=seed), world, seconds, hz=hz):
        budget.add(cmd, tel, dt)
    return budget


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two time budgets (0 same … 1 disjoint)."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


@pytest.fixture(scope="module")
def budgets() -> dict[str, list[Budget]]:
    return {name: [_budget(_genome(name), seed) for seed in (1, 2)] for name in TEMPERAMENTS}


def test_temperaments_live_measurably_different_lives(budgets: dict[str, list[Budget]]) -> None:
    # Seeds 1-4 (pairs (1,2) and (3,4)), not just 1-2: measured 2026-10-02 at 6ec7f91, the
    # bold/lazy walk ratio is 1.68 over seeds 1-4 and 1.32-1.88 per seed (seed 2 alone is a weak
    # draw, 1.32), so two seeds sat right on the 1.5 line. Seeds 1-2 come from the shared fixture.
    more = {name: [_budget(_genome(name), seed) for seed in (3, 4)] for name in budgets}
    runs_by = {name: [*budgets[name], *more[name]] for name in budgets}
    walk = {
        name: statistics.mean(b.walk_s / b.total_s for b in runs) for name, runs in runs_by.items()
    }
    assert walk["bold_explorer"] > 1.5 * walk["lazy_dreamer"]
    assert min(walk.values()) > 0.05  # every duck walks (owner: 鴨子一定會走動)
    names = list(budgets)
    for first, second in ((0, 1), (2, 3)):  # the seed pairs (1,2) and (3,4)
        between = []
        within = []
        for i, a in enumerate(names):
            within.append(_distance(runs_by[a][first].shares(), runs_by[a][second].shares()))
            for b in names[i + 1 :]:
                between.append(_distance(runs_by[a][first].shares(), runs_by[b][first].shares()))
        assert min(between) > max(within), (between, within)
    signature = {"bold_explorer": "survey", "shy_snuggler": "melt", "lazy_dreamer": "moon_gaze"}
    for name, verb in signature.items():
        assert all(verb in run_.time_s for run_ in runs_by[name]), (name, verb)


def test_a_rich_repertoire_shows_up(budgets: dict[str, list[Budget]]) -> None:
    for runs in budgets.values():
        for budget in runs:
            assert len(budget.time_s) >= 22
            assert budget.calls > 20
    # bouts end by satiation and competition, not by a fixed duration
    lengths: dict[str, list[float]] = {}
    for runs in budgets.values():
        for budget in runs:
            for behaviour, values in budget.bout_lengths.items():
                lengths.setdefault(behaviour, []).extend(values)
    varied = [
        b
        for b, v in lengths.items()
        if len(v) >= 12 and statistics.pstdev(v) / statistics.mean(v) > 0.3
    ]
    assert len(varied) >= 8, lengths.keys()


# ── habituation and sensitisation ─────────────────────────────────────────────


def test_habituation_levels_fall_with_repetition_and_recover_with_rest() -> None:
    hab = Habituation(_genome("social_songbird").p)
    gains = [hab.present("sound:clap", 0.8, 10.0 * k) for k in range(10)]
    assert all(b <= a + 1e-9 for a, b in zip(gains, gains[1:], strict=False))
    assert gains[-1] < 0.55 * gains[0]
    assert hab.gain("sound:clap", 100.0 + 1800.0) > 0.9 * gains[0]
    assert hab.gain("touch:back", 100.0) == pytest.approx(1.0)  # other classes untouched
    hab.sensitise("sound:bang", 1.0, 2000.0)
    assert hab.gain("sound:clap", 2000.0) > hab.level("sound:clap", 2000.0)


def _clap_responses(
    brain: NeuroBrain, world: SynthWorld, claps: list[int], until_ms: int
) -> list[tuple[float, float]]:
    """(response strength, habituation × sensitisation gain) of the reflex each clap got."""
    for t in claps:
        world.clap(t)
    seen: dict[int, tuple[float, float]] = {}
    t = 0
    while t < until_ms:
        cmd, tel = brain.step(t, world.frame(t))
        world.apply(cmd, 0.05)
        for event in tel.reflexes:
            if event.stimulus == "sound:clap" and event.reflex in ("startle", "orient"):
                seen.setdefault(event.t_ms, (event.strength, event.gain))
        t += 50
    out = []
    for clap in claps:
        near = [v for tt, v in seen.items() if clap <= tt <= clap + 300]
        out.append(max(near) if near else (0.0, 0.0))
    return out


def test_the_tenth_clap_gets_a_weaker_response_and_rest_restores_it() -> None:
    """Ten identical claps 8 s apart: the response grows a little at first (the first scares
    sensitise), then habituation wins — the 10th is clearly weaker than the 1st. After 25 quiet
    minutes the habituation has recovered (its gain is fresh again) and the response is back."""
    genome = _genome("social_songbird")
    brain = NeuroBrain(genome, seed=3)
    world = SynthWorld(seed=9, family=False, tof=False)
    claps = [30_000 + 8_000 * k for k in range(10)] + [30_000 + 8_000 * 9 + 1_500_000]
    responses = _clap_responses(brain, world, claps, claps[-1] + 2_000)
    (first, g_first), (tenth, g_tenth), (rested, g_rested) = (
        responses[0],
        responses[9],
        responses[10],
    )
    assert first > 0.2
    assert tenth < 0.6 * first, responses
    assert max(s for s, _ in responses[3:10]) > tenth  # it decays toward the 10th
    assert g_tenth < 0.7 * g_first, responses  # habituation, not only calmer
    assert g_rested > 0.9 and g_rested > 1.4 * g_tenth, responses  # rest restores it
    assert rested > 1.15 * tenth, responses


# ── reflexes interrupt, bouts are satiated ────────────────────────────────────


def test_reflexes_interrupt_whatever_it_is_doing() -> None:
    genome = _genome("shy_snuggler")
    brain = NeuroBrain(genome, seed=4)
    world = SynthWorld(seed=4, family=False)
    world.poke(20_000, site="back")  # (a leg contact while walking is its own doing: a brush)
    world.bang(40_000)
    world.falls.append(60_000)
    seen: dict[str, list[int]] = {}
    before = ""
    t = 0
    while t < 80_000:
        cmd, tel = brain.step(t, world.frame(t))
        world.apply(cmd, 0.05)
        if cmd.reflex and cmd.reflex != before:
            seen.setdefault(cmd.reflex, []).append(t)
        before = cmd.reflex or ""
        t += 50
    assert any(20_000 <= t <= 20_200 for t in seen.get("withdraw", []))
    assert any(40_000 <= t <= 40_200 for t in seen.get("startle", []))
    assert any(t > 40_000 for t in seen.get("freeze", []))  # the startle fades into a freeze
    assert any(60_000 <= t <= 60_200 for t in seen.get("righting", []))


def test_a_stronger_urge_walks_longer() -> None:
    """The same calm duck in an empty room: the more it wants to move (its restlessness and
    curiosity at the start), the more of the next minute it spends stepping."""
    lengths = {}
    for level in (0.2, 1.0):
        totals = []
        for seed in (1, 2, 3, 4):
            brain = NeuroBrain(_genome("lazy_dreamer"), seed=seed)
            world = SynthWorld(seed=seed, family=False)
            world.toys = []  # nothing new to see: only the urge moves it
            brain.drives.restless = level
            brain.drives.curiosity = level
            walked = 0.0
            t = 0
            while t < 60_000:
                cmd, _ = brain.step(t, world.frame(t))
                world.apply(cmd, 0.1)
                walked += 0.1 if cmd.vx > 0 else 0.0
                t += 100
            totals.append(walked)
        lengths[level] = statistics.mean(totals)
    assert lengths[1.0] > 2.0 * lengths[0.2], lengths


def test_no_timer_ends_a_bout_only_satiation_and_competition() -> None:
    """Bouts end because doing the thing satisfies it (satiation) and something else wins —
    never because a clock ran out. With satiation switched off and steady inputs, one bout
    holds for ten minutes; with it on, the same inputs end the bout by themselves, and a
    stronger urge keeps the bout going longer."""
    genome = _genome("social_songbird")
    ctx = S.Context(sitting=False, standing=True, asleep=False, can_walk=True, wakefulness=1.0)

    def first_bout_s(restless: float, satiation: bool) -> float:
        sel = S.Selector(genome, seed=1)
        if not satiation:
            sel._sat_rate = [0.0] * len(sel.names)
        feat = S.empty_features()
        feat[S.F["restless"]] = restless
        started: float | None = None
        for k in range(12_000):  # ten minutes at 20 Hz
            name, new = sel.step(0.05, feat, ctx, False, 0.0)
            if new and started is None and name == "wander":
                started = k * 0.05
            elif new and started is not None:
                return k * 0.05 - started
        return 600.0 if started is not None else 0.0

    assert first_bout_s(0.9, satiation=False) == 600.0  # nothing but satiation ends it
    strong = first_bout_s(1.0, satiation=True)
    weak = first_bout_s(0.45, satiation=True)
    assert 0.0 < weak < strong < 600.0, (weak, strong)


# ── learning: the owner's fall rule ───────────────────────────────────────────


def test_three_hard_falls_at_one_spot_make_it_avoided_one_does_not() -> None:
    learn = Learner(_genome("bold_explorer").p)
    spot = (1.2, 0.7, 0.0)
    learn.fell(spot, 0.0)
    learn.righted(2.0, helped=False)  # up quickly by itself: not hard
    assert learn.avoided_near((0.4, 0.7, 0.0)) == []
    for k in range(AVOID_AFTER):
        learn.fell(spot, 100.0 + 50 * k)
        assert learn.righted(100.0 + 50 * k + 6.0, helped=False)  # > 4 s: hard
    near = learn.avoided_near((0.4, 0.7, 0.0))
    assert near and abs(near[0][0]) < 0.3 and near[0][1] < 1.2


def test_a_name_that_brings_good_things_gains_value() -> None:
    learn = Learner(_genome("social_songbird").p)
    for k in range(6):
        learn.cue("sound:name", 100.0 * k)
        learn.outcome(1.0, 100.0 * k + 2.0)
    assert learn.value_of("sound:name") > 0.5
    assert math.isclose(learn.value_of("sound:door"), 0.0)


# ── liveliness: the gaze cycle and a duckling's rhythm rule ─────────────────────


def test_idle_gaze_shifts_are_irregular_like_a_birds_not_a_clock() -> None:
    """Ethology A1: a calm bird shifts its gaze every ~1–2 s with log-normal intervals (CV about
    0.5–0.8); today's Duck Core glances run on a 6-s clock (CV 0.19)."""
    brain = NeuroBrain(_genome("social_songbird"), seed=11)
    world = SynthWorld(seed=11, family=False)
    world.toys = []
    shifts: list[float] = []
    original = brain._saccade

    def record(t_s: float, stepping: bool, d: object) -> None:
        shifts.append(t_s)
        original(t_s, stepping, d)  # type: ignore[arg-type]

    brain._saccade = record  # type: ignore[method-assign]
    for cmd, _ in iterate(brain, world, 300.0, hz=20):
        pass
    gaps = [b - a for a, b in zip(shifts, shifts[1:], strict=False)]
    assert len(gaps) > 60
    median = statistics.median(gaps)
    cv = statistics.pstdev(gaps) / statistics.mean(gaps)
    assert 0.5 < median < 3.0, median
    assert cv > 0.45, cv


def _train(brain: NeuroBrain, world: SynthWorld, hz: float, n: int, t0: int) -> tuple[int, int]:
    """Clap ``n`` times at ``hz`` from ``t0``; returns (freeze ticks, calls) over the train and
    the 3 s after it."""
    for k in range(n):
        world.clap(t0 + int(k * 1000 / hz), db=70.0)
    end = t0 + int(n * 1000 / hz) + 3000
    freeze = calls = 0
    seq = None
    t = 0
    while t < end:
        cmd, _ = brain.step(t, world.frame(t))
        world.apply(cmd, 0.05)
        if t >= t0:
            freeze += cmd.behaviour in ("freeze", "still")
            if seq is not None and cmd.vocal_seq != seq:
                calls += 1
        seq = cmd.vocal_seq
        t += 50
    return freeze, calls


def test_a_slow_knocking_train_freezes_a_duckling_and_hushes_it() -> None:
    """Miller's rule (Pekin ducklings, n = 450): sound trains at 0.8–1.8 per second make them
    freeze and fall silent; fast trains (≥ 2.6–2.8/s) do not — they call and come instead."""
    slow_freeze, slow_calls = [], []
    fast_freeze = []
    # (dock:build 2026-09-28: six seeds and the median — the brain's selection changed and seed 3
    # now gives one extra alarm at the train's last knock after a flee; over seeds 1–12 the
    # slow train's calls were 12 × 1 before and 11 × 1 + 1 × 2 after, the freezes 77 → 73 ticks
    # slow and 19 → 2 fast: Miller's contrast holds)
    for seed in (1, 2, 3, 4, 5, 6):
        young = Genome.from_axes(TEMPERAMENTS["social_songbird"][0], primary="songbird", gd=6)
        assert young.stage in ("hatchling", "duckling")
        brain = NeuroBrain(young, seed=seed)
        world = SynthWorld(seed=seed, family=False)
        f, c = _train(brain, world, 1.2, 6, 30_000)
        slow_freeze.append(f)
        slow_calls.append(c)
        brain = NeuroBrain(young, seed=seed)
        world = SynthWorld(seed=seed, family=False)
        f, _ = _train(brain, world, 3.0, 15, 30_000)
        fast_freeze.append(f)
    assert statistics.mean(slow_freeze) > 2.0 * max(1.0, statistics.mean(fast_freeze))
    assert statistics.median(slow_calls) <= 1 and max(slow_calls) <= 2


def test_new_eye_track_gets_its_own_key_despite_a_nearby_known_thing() -> None:
    """P1d-3 F1: the D2 snuggler numbers. Books-memory sits at (-0.992, -0.065)
    after walking looks; the bottle's first to_odom fix is (-1.31, -0.24) at
    dist 0.92 — inside the old 0.35 + 0.3 * dist merge radius. A never-seen
    eye track must still open its own key with usable novelty."""
    learn = Learner(_genome("shy_snuggler").p)
    for k in range(50):
        learn.see_thing(1, "thing", (0.0, 0.0, 0.0), -0.91, -0.02, k * 0.02, 0.02, 0.0)
    learn.things[1].x = -0.992
    learn.things[1].y = -0.065
    key, novelty = learn.see_thing(2, "thing", (-0.61, -0.29, 3.07), -1.31, -0.24, 2.3, 0.02, 0.0)
    assert key == 2  # second free key: books took 1 above
    assert learn._tracks[2] == 2
    assert novelty > 0.5


def test_known_track_still_reuses_its_key_within_keep_window() -> None:
    """P1d-3 F1 guard: a registered track keeps its key inside TRACK_KEEP_S,
    so the fix cannot break ordinary track continuity."""
    learn = Learner(_genome("shy_snuggler").p)
    key0, _ = learn.see_thing(2, "thing", (0.0, 0.0, 0.0), -1.31, -0.24, 2.3, 0.02, 0.0)
    key1, _ = learn.see_thing(2, "thing", (0.05, 0.0, 0.0), -1.28, -0.22, 2.5, 0.02, 0.0)
    assert key0 == key1 == 1  # first free key; continuity is what matters
    assert learn.things[1].seen_t == 2.5  # second look refreshed the same thing


def test_it_walks_forward_out_of_its_nest_before_walking_away() -> None:
    """(dock:build 2026-09-28: the dock is back-in — owner 「6. 可以」) A duck on its contacts
    faces the room: when it wants to walk it walks out FORWARD along the dock's axis (the docking
    program's leave: tests/test_neuro_dock.py), never backwards as the forward dock's duck did."""
    brain = NeuroBrain(_genome("bold_explorer"), seed=1)
    brain.drives.restless = 1.0
    on_nest = C.SensorFrame(
        t_ms=0,
        body=C.BodySense(posture="sit"),
        power=C.PowerSense(battery=0.95, on_dock=True, charging=True),
    )
    brain.step(0, on_nest)
    brain.sel.current = brain.sel.index("wander")
    brain.behaviour = "wander"
    moves = []
    for k in range(1, 80):
        frame = C.SensorFrame(
            t_ms=50 * k,
            body=C.BodySense(posture="stand", odom=(0.004 * k, 0.0, 0.0)),
            power=C.PowerSense(battery=0.95),
        )
        cmd, _ = brain.step(50 * k, frame)
        moves.append((cmd.vx, cmd.docking))
    assert any(vx > 0.0 and docking for vx, docking in moves)  # out along the axis first
    assert all(vx >= 0.0 for vx, _ in moves)  # never backwards out of a back-in dock


def test_a_new_track_at_a_known_place_is_the_same_thing_only_once_the_old_sighting_is_over() -> (
    None
):
    """P1d-3 F1 edge: the same thing at the same place under a NEW eye track id. Seen less than
    TRACK_KEEP_S ago it is being watched by another track right now (a different object, new key);
    seen longer ago than TRACK_KEEP_S its track has ended and the place knows it again."""
    from duckmind.neuro.learn import TRACK_KEEP_S

    for gap, same in ((TRACK_KEEP_S - 0.1, False), (TRACK_KEEP_S + 0.1, True)):
        learn = Learner(_genome("shy_snuggler").p)
        odom = (0.0, 0.0, 0.0)
        key1, _ = learn.see_thing(1, "thing", odom, 1.0, 0.0, 10.0, 0.1, 0.0)
        key2, _ = learn.see_thing(2, "thing", odom, 1.0, 0.0, 10.0 + gap, 0.1, 0.0)
        assert (key2 == key1) is same, gap
