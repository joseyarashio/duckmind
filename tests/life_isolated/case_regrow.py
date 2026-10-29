"""Growth G3: the duck grows in its sleep (docs/design/GROWTH-G3-2026-10-11.md; runs in a child
process, see tests/test_life_isolated.py). The brain-level cases: a regrown brain is what a brain
built for the new genome is, it keeps everything it lived, it is never regrown awake; and what the
family's care does to a genome."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld)
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.life import regrow as rg
from duckmind.neuro import select as S
from duckmind.neuro.genome import (
    BOND_ACTS,
    BOND_HATCH,
    BOND_SOCIAL,
    PREF_FACTOR_RANGE,
    Care,
    Genome,
    pref_factor,
)
from duckmind.neuro.integration import (
    IntegratedBrain,
    RegrowRefused,
    install_default_skills,
    make_brain,
)
from duckmind.neuro.synth import SynthWorld

SEED = 4107

# what a regrown brain may differ in from one built for the same genome: the selector's noise (its
# stream goes on from where it was; a fresh one starts again) and nothing else
STATE_ONLY = ("sel._noise", "sel._m_noise", "sel._stream")


def _attrs(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if hasattr(obj, "__dict__"):
        out.update(vars(obj))
    for cls in type(obj).__mro__:
        for slot in getattr(cls, "__slots__", ()) or ():
            if isinstance(slot, str) and hasattr(obj, slot):
                out[slot] = getattr(obj, slot)
    return out


def deep_diff(a: Any, b: Any, path: str, out: list[str], seen: set[tuple[int, int]]) -> None:
    """Every place two object graphs differ (paths), floats and ints and strings by value."""
    if a is b or len(out) > 200:
        return
    if isinstance(a, float) and isinstance(b, float):
        if a != b and not (math.isnan(a) and math.isnan(b)):
            out.append(path)
        return
    if type(a) is not type(b):
        out.append(path)
        return
    if isinstance(a, (int, str, bytes, bool, type(None))):
        if a != b:
            out.append(path)
        return
    if callable(a) and not hasattr(a, "__dict__"):
        return
    key = (id(a), id(b))
    if key in seen:
        return
    seen.add(key)
    if isinstance(a, dict):
        if set(a) != set(b):
            out.append(path + "{keys}")
        for k in a:
            if k in b:
                deep_diff(a[k], b[k], f"{path}[{k!r}]", out, seen)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(path + "{len}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            deep_diff(x, y, f"{path}[{i}]", out, seen)
    elif isinstance(a, (set, frozenset)):
        if a != b:
            out.append(path + "{set}")
    elif hasattr(a, "tolist"):
        return
    else:
        da, db = _attrs(a), _attrs(b)
        for k in da:
            if k in db:
                deep_diff(da[k], db[k], f"{path}.{k}", out, seen)


def _asleep(genome: Genome) -> IntegratedBrain:
    """A brain that has not stepped, put to sleep (the sleeping act holds the body)."""
    brain = make_brain(genome, SEED)
    index = brain.sel.names.index("doze")
    brain.sel.current = index
    brain.behaviour = "doze"
    return brain


def _g(gd: int, branch: str | None = "auto") -> Genome:
    return Genome.from_seed(SEED, gd=gd, branch=branch)


CASES = (
    ("hatchling to duckling", _g(3), _g(20)),
    ("duckling to juvenile with its branch", _g(47, None), _g(60, "explorer")),
    ("a branch changes", _g(60, "explorer"), _g(60, "snuggler")),
    (
        "the care of a month",
        _g(60, "explorer"),
        replace(
            _g(61, "explorer"), care=Care(bond=5000, prefs=(("wander", 5600), ("snuggle", 4400)))
        ),
    ),
)


@pytest.mark.parametrize(("label", "before", "after"), CASES, ids=[c[0] for c in CASES])
def test_a_regrown_sleeping_brain_is_a_brain_built_for_the_new_genome(
    label: str, before: Genome, after: Genome
) -> None:
    install_default_skills()
    grown = _asleep(before)
    grown.regrow(after)
    reference = _asleep(after)
    differ: list[str] = []
    deep_diff(grown, reference, "brain", differ, set())
    other = [p for p in differ if not any(f".{state}" in p for state in STATE_ONLY)]
    assert other == [], f"{label}: a genome-derived holder was left behind: {other[:12]}"


def test_a_brain_is_never_regrown_awake() -> None:
    install_default_skills()
    awake = make_brain(_g(20), SEED)
    assert awake.behaviour not in S.SLEEP
    before = awake.genome
    with pytest.raises(RegrowRefused):
        awake.regrow(_g(21))
    assert awake.genome is before  # nothing was touched


def test_what_it_lived_stays_when_it_grows() -> None:
    """A brain that has really lived (the synthetic room, hours of brain time) and fallen asleep:
    its learner, habituation, drives, clocks and the act that holds the body are the same after."""
    install_default_skills()
    world = SynthWorld(seed=3, family=False, day_minute0=0.0)
    brain = make_brain(_g(20), SEED)
    t_ms = 0
    calm_s = 0.0
    while calm_s < 30.0 and t_ms < 900_000:
        frame = world.frame(t_ms)
        cmd, _tel = brain.step(t_ms, frame)
        world.apply(cmd, 0.1)
        t_ms += 100
        calm_s = calm_s + 0.1 if rg.calm(brain, frame, t_ms, cmd) is None else 0.0
    assert calm_s >= 30.0, "the duck never slept calmly in 15 minutes of its room"
    t_s = (t_ms - brain._t0) / 1000.0
    kept = {
        "learner": brain.learn.to_state(t_s),
        "hab": brain.hab.to_state(t_s),
        "drives": brain.drives.to_state(),
        "behaviour": brain.behaviour,
        "bout": brain.bout,
        "counts": dict(brain.counts),
        "t": (brain._t0, brain._t_prev),
        "act": dict(zip(brain.sel.names, brain.sel.act, strict=True)),
    }
    change = brain.regrow(_g(60, "explorer"))
    assert change["stage"] == ["duckling", "juvenile"] and change["branch"] == [None, "explorer"]
    after = {
        "learner": brain.learn.to_state(t_s),
        "hab": brain.hab.to_state(t_s),
        "drives": brain.drives.to_state(),
        "behaviour": brain.behaviour,
        "bout": brain.bout,
        "counts": dict(brain.counts),
        "t": (brain._t0, brain._t_prev),
        "act": {
            n: a for n, a in zip(brain.sel.names, brain.sel.act, strict=True) if n in kept["act"]
        },
    }
    assert after == kept
    assert brain.sel.names[brain.sel.current] == kept["behaviour"]
    # and it goes on living: the first ticks after the swap are the ticks of a sleeping duck
    asleep_ticks = 0
    for _ in range(100):
        frame = world.frame(t_ms)
        cmd, _tel = brain.step(t_ms, frame)
        world.apply(cmd, 0.1)
        t_ms += 100
        asleep_ticks += int(cmd.asleep)
    assert asleep_ticks >= 20


def test_neutral_care_changes_nothing_not_a_bit() -> None:
    install_default_skills()
    plain = _g(60, "explorer")
    carried = replace(plain, care=Care())
    assert carried.with_care_applied() is carried  # (nothing to apply)
    a, b = make_brain(plain, SEED), make_brain(carried, SEED)
    assert dataclasses.asdict(a.genome.p) == dataclasses.asdict(b.genome.p)
    assert a.genome.gains == b.genome.gains and a.genome.care_applied is False


def test_the_bond_and_the_preferences_reach_the_genome_by_the_rules_stated() -> None:
    install_default_skills()
    base = _g(60, "explorer")
    plain = make_brain(base, SEED).genome
    bonded = make_brain(replace(base, care=Care(bond=6000)), SEED).genome  # u = 0.5
    u = Care(bond=6000).bond_unit
    assert u == 0.5
    assert bonded.p.social_tau_s == pytest.approx(plain.p.social_tau_s / (1.0 + BOND_SOCIAL * u))
    assert bonded.p.greet_reflex > plain.p.greet_reflex
    for behaviour in BOND_ACTS:
        if behaviour in plain.gains:
            assert bonded.gains[behaviour] == pytest.approx(
                plain.gains[behaviour] * (1 + BOND_SOCIAL * u)
            )
    assert bonded.gains["wander"] == plain.gains["wander"]  # (nothing else moved)
    # a preference: the act's salience, up or down, bounded
    liked = make_brain(replace(base, care=Care(prefs=(("wander", 5600), ("snuggle", 4400)))), SEED)
    assert liked.genome.gains["wander"] == pytest.approx(plain.gains["wander"] * pref_factor(5600))
    assert liked.genome.gains["snuggle"] < plain.gains["snuggle"]
    assert liked.genome.gains["preen"] == plain.gains["preen"]
    low, high = PREF_FACTOR_RANGE
    assert pref_factor(5000) == 1.0 and pref_factor(0) == low and pref_factor(10_000) == high
    # a genome carries its care once: applying it again is the same genome
    again = liked.genome.with_care_applied()
    assert again is liked.genome


def test_a_basis_survives_the_snapshot_and_builds_the_genome_of_the_state_it_came_from() -> None:
    install_default_skills()
    session = Session(
        Config(seed=SEED, name="蛋黃", start="2026-10-10T15:00:00+08:00", household="legacy")
    )
    for _ in range(60):
        dispatch(session, "advance", {"seconds": 86400})
    state = session.core.s
    state.bond_owner = 4700
    state.prefs["household.verb:wander"] = 5333
    state.prefs["household.verb:preen"] = 5000  # (neutral: not part of the care)
    state.prefs["thing:3"] = 6000  # (not an act: not part of the care)
    basis = rg.basis_of(state, branch=state.branch)
    static = rg.Static.of(state)
    assert basis.care == Care(bond=4700, prefs=(("wander", 5333),))
    again = rg.Basis.from_json(basis.to_json())
    assert again is not None and again.same_as(basis)
    direct = Genome.from_core_state(state)
    built = static.genome(basis)
    assert built.care == direct.care and built.gd == direct.gd and built.branch == direct.branch
    assert dataclasses.asdict(built.p) == dataclasses.asdict(direct.p)
    assert built.gains == direct.gains and built.unlocked == direct.unlocked
    # a stamp of another revision is not trusted
    stamp = basis.to_json()
    stamp["rev"] = "growth-g0"
    assert rg.Basis.from_json(stamp) is None and rg.Basis.from_json(None) is None
    assert rg.Basis.from_json({"rev": rg.GROWTH_REV, "gd": "x"}) is None


def test_the_hatchlings_care_is_neutral_so_its_genome_is_the_seeds() -> None:
    install_default_skills()
    session = Session(
        Config(seed=SEED, name="蛋黃", start="2026-10-10T15:00:00+08:00", household="legacy")
    )
    state = session.core.s
    assert Care.from_state(state) == Care(bond=BOND_HATCH)
    genome = Genome.from_core_state(state)
    reference = Genome.from_seed(SEED, gd=0)
    assert dataclasses.asdict(genome.p) == dataclasses.asdict(reference.p)
    assert genome.gains == reference.gains


# ── the life on a server: the offer, the sleep, the snapshot, the restart ──────────────────────

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_restart import BIRTH, Life, Wall  # noqa: E402
from test_neuro_live import robot_state  # noqa: E402


def _one_day(life: Life) -> None:
    """A day of the calendar passes with the duck awake (the minute-by-minute catch-up)."""
    for _ in range(24 * 15):
        life.wall.t += 240.0
        with life.state.lock:
            life.state.grow()


def _tick(life: Life, seconds: float, *, policy: str = "sit", every_s: float = 1.0) -> None:
    """The driver's ticks (50 Hz, no body); the host's watch looks for an offer once a second."""
    n = int(seconds / 0.02)
    start = getattr(life, "_k", 0)
    for k in range(start, start + n):
        life.driver.tick(robot_state(k * 0.02, policy=policy))
        if k % int(every_s / 0.02) == 0:
            life.host.regrow_watch()
    life._k = start + n  # type: ignore[attr-defined]


def _spy(life: Life) -> list[str]:
    """What the brain was doing each time it was regrown."""
    seen: list[str] = []
    real = life.brain.regrow

    def regrow(genome: Genome) -> dict[str, Any]:
        seen.append(str(life.brain.behaviour))
        return real(genome)

    life.brain.regrow = regrow  # type: ignore[method-assign]
    return seen


def test_a_growth_day_is_grown_into_only_while_it_sleeps(tmp_path: Path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH))
    life.host.start()
    try:
        seen = _spy(life)
        voiced: list[str] = []
        life.host.refresh_voice = lambda why="night", **_: voiced.append(why) or "unavailable"  # type: ignore[method-assign]
        assert life.host.regrower is not None and life.host.regrower.applied.gd == 0
        _tick(life, 20.0, policy="stand")  # a day's work, awake: no offer, nothing happens
        assert seen == [] and not life.host.regrower.due
        _one_day(life)
        life.host.regrow_watch()
        assert life.host.regrower.due and life.core.s.gd == 1
        _tick(life, 25.0, policy="stand")  # the day clock says 10:00: it is awake, so it waits
        assert seen == [] and life.host.status()["regrow"]["waiting_for"] == "awake"
        life.driver.set_day(120)  # night
        _tick(life, 120.0)
        assert life.host.regrows == 1, life.host.status()["regrow"]
        assert seen and all(b in S.SLEEP for b in seen)  # (every swap was made asleep)
        view = life.host.status()["regrow"]
        assert view["count"] == 1 and view["applied_gd"] == 1 and view["due"] is False
        assert view["last"]["why"] == "night" and view["last"]["gd"] == [0, 1]
        assert view["genome"]["gd"] == 1 and life.driver.lanes.genome["gd"] == 1
        assert life.driver.spec.duck == "4107:1" and life.driver.spec.gd == 1
        assert "regrow" in voiced  # (growth G5: the lineage voice is looked at again)
        for _ in range(100):  # the watch thread writes the snapshot that makes it the duck's
            saved = json.loads(life.record.neural_path.read_text()).get("regrow") or {}
            if saved.get("gd") == 1:
                break
            time.sleep(0.1)
        assert saved.get("gd") == 1 and saved.get("regrows") == 1
    finally:
        life.host._stop.set()


def test_a_fast_forward_over_many_days_is_one_regrow_to_the_latest(tmp_path: Path) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH))
    life.host.start()
    try:
        seen = _spy(life)
        for _ in range(7):
            _one_day(life)
        life.host.regrow_watch()
        assert life.core.s.gd == 7
        life.driver.set_day(120)
        _tick(life, 120.0)
        assert life.host.regrows == 1 and len(seen) == 1
        change = life.host.regrow_history[-1]
        assert change["gd"] == [0, 7]
        assert life.host.regrower.applied.gd == 7 and "wander" in life.brain.genome.unlocked
        assert change["unlocked"], "seven days of growth: the first steps are unlocked"
    finally:
        life.host._stop.set()


def test_after_a_regrow_a_restart_builds_the_same_genome_not_the_one_of_the_state_now(
    tmp_path: Path,
) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.host.start()
    try:
        _one_day(first)
        first.driver.set_day(120)
        _tick(first, 120.0)
        assert first.host.regrows == 1
        assert first.host.snapshot_now("regrow")
        with first.state.lock:
            first.state.save()  # (a server saves Duck Core every second)
        stamp = json.loads((folder / "neural.json").read_text())["regrow"]
        assert stamp["gd"] == 1 and stamp["rev"] == rg.GROWTH_REV and stamp["regrows"] == 1
        grown = first.brain.genome
    finally:
        first.host._stop.set()
    # the stamp says the bond was 6000 when it grew (as if the day's care had moved Duck Core on
    # since): the restart of the same growth day gives back the genome it had, not today's state
    data = json.loads((folder / "neural.json").read_text())
    data["regrow"]["care"] = {"bond": 6000, "prefs": {"wander": 5400}}
    (folder / "neural.json").write_text(json.dumps(data))
    second = Life(folder, wall)
    assert second.host.build_basis()[1] in ("snapshot", "kept")
    again = second.brain.genome
    assert again.gd == grown.gd == 1 and again.care == Care(bond=6000, prefs=(("wander", 5400),))
    assert again.p.social_tau_s < grown.p.social_tau_s  # (it is the stamp's care that is in force)
    # …and the same stamp without the care edit is the very genome of the first run
    data["regrow"]["care"] = {"bond": BOND_HATCH, "prefs": {}}
    (folder / "neural.json").write_text(json.dumps(data))
    third = Life(folder, wall)
    assert dataclasses.asdict(third.brain.genome.p) == dataclasses.asdict(grown.p)
    assert third.brain.genome.gains == grown.gains and third.brain.genome.unlocked == grown.unlocked


def test_a_night_missed_while_the_server_was_off_is_grown_into_at_the_boot(tmp_path: Path) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.host.start()
    try:
        _one_day(first)  # it is on for a day (gd 1), the mind is saved with gd 0's body
        first.host.snapshot_now("night")
        with first.state.lock:
            first.state.save()
    finally:
        first.host._stop.set()
    stamp = json.loads((folder / "neural.json").read_text())["regrow"]
    assert stamp["gd"] == 0
    first.host.stop()
    wall.t += 3 * 86400.0
    second = Life(folder, wall)
    gd = second.core.s.gd
    assert gd >= 1 and second.brain.genome.gd == gd  # built at the day it has, not the stamp's
    assert second.host.build_basis()[1] in ("kept", "state")


def test_the_record_that_pins_the_branch_follows_the_record_not_an_old_stamp(
    tmp_path: Path,
) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.host.start()
    first.host._stop.set()
    first.host.snapshot_now("night")
    data = json.loads((folder / "neural.json").read_text())
    data["regrow"]["branch"] = "dreamer"  # a stamp from before the record's pin
    (folder / "neural.json").write_text(json.dumps(data))
    second = Life(folder, wall)
    assert second.host.record.branch_from_core is False
    assert second.brain.genome.branch in (
        None,
        second.host.nature_branch(),
    )  # not the stamp's dreamer


def test_a_demo_duck_on_screen_is_not_regrown_and_the_life_duck_returns_at_todays_day(
    tmp_path: Path,
) -> None:
    life = Life(tmp_path / "duck", Wall(BIRTH))
    life.host.start()
    try:
        life.driver.switch("1728:20")
        assert not life.driver.spec.life
        _one_day(life)
        life.driver.set_day(120)
        _tick(life, 60.0)
        assert life.host.regrows == 0
        life.driver.switch(life.host.spec_string())  # back to 蛋黃: built at today's growth day
        assert life.driver.spec.life and life.driver.lanes.genome["gd"] == life.core.s.gd == 1
    finally:
        life.host._stop.set()


def test_a_snapshot_from_before_g3_has_no_basis_and_the_duck_boots_from_duck_cores_state(
    tmp_path: Path,
) -> None:
    """G1 and G2 records (and the copy of 蛋黃) have a neural.json without a ``regrow`` section: they
    load as they always did; the genome is built from Duck Core's state, and the next night grows
    the body from there."""
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.host.start()
    try:
        _one_day(first)
        first.host.snapshot_now("night")
        with first.state.lock:
            first.state.save()
    finally:
        first.host._stop.set()
    data = json.loads((folder / "neural.json").read_text())
    assert "regrow" in data
    del data["regrow"]  # (what G1 and G2 wrote)
    (folder / "neural.json").write_text(json.dumps(data))
    second = Life(folder, wall)
    assert second.host._stamp is None and second.report is not None  # (the mind came back)
    assert second.host.build_basis()[1] in ("kept", "state")
    assert second.brain.genome.gd == second.core.s.gd == 1
    assert second.host.regrower is not None and not second.host.regrower.due


def test_a_growth_day_that_is_not_the_stamps_with_a_foreign_revision_is_not_trusted(
    tmp_path: Path,
) -> None:
    folder, wall = tmp_path / "duck", Wall(BIRTH)
    first = Life(folder, wall)
    first.host.start()
    first.host._stop.set()
    first.host.snapshot_now("night")
    data = json.loads((folder / "neural.json").read_text())
    data["regrow"]["rev"] = "growth-g9"
    data["regrow"]["care"] = {"bond": 9000, "prefs": {}}
    (folder / "neural.json").write_text(json.dumps(data))
    second = Life(folder, wall)
    assert second.host._stamp is None
    assert second.brain.genome.care == Care()  # (not the stamp's bond)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "tools" / "growth").is_dir(),
    reason="tools/growth (the G3 gate's harness) is not in this checkout",
)
def test_the_headless_harness_grows_a_duck_in_its_sleep_a_week_and_the_hatchling_cannot_walk() -> (
    None
):
    """The G3 gate's own harness, a week long: nightly regrows are made asleep and keep what the duck
    lived; the body is a hatchling that cannot walk until its first steps unlock (growth day 5 at its
    pace), and then it walks."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tools" / "growth"))
    import regrow_harness as harness

    life = harness.run_life(2, "attentive", days=7, care_from=1, care_days=5, do_exam=False)
    assert life["regrows"] >= 5 and life["missed_nights"] == 0
    assert life["audit_all_asleep"] and life["audit_all_state_kept"]
    walked = [row["path_m"] for row in life["daily"]]
    first_steps = life["daily"][0]["unlocked"]
    assert walked[0] == 0.0 and walked[1] == 0.0  # a hatchling does not walk
    assert max(walked[4:]) > 0.0, walked  # …and after its first steps it does
    assert life["daily"][-1]["unlocked"] > first_steps


def test_a_play_skill_the_duck_ended_long_ago_does_not_keep_it_from_growing() -> None:
    """``PlayMind.out`` keeps the last play act's one-shot; only ``brain.play_out`` (this tick's
    wish) says a play skill is going on. A sleeping duck that once kicked a ball is calm."""
    install_default_skills()
    from dataclasses import replace as rep

    from duckmind.neuro.skill_common import PlayOut

    world = SynthWorld(seed=3, family=False, day_minute0=0.0)
    brain = _asleep(_g(60, "explorer"))
    frame = world.frame(0)
    brain._t0, brain._t_prev = 0, 0
    brain.play.out = rep(PlayOut(), skill="kick_left")  # stale, from long ago
    assert rg.calm(brain, frame, 100_000) is None
    brain.play_out = rep(PlayOut(), skill="kick_left")  # …but one going on now is not calm
    assert rg.calm(brain, frame, 100_000) == "play"
