"""duck-body's skill guard and wear budget (:mod:`duckmind.skilltrain.guard`)."""

from __future__ import annotations

import statistics
from typing import Any

from duckmind.skilltrain import guard as G

ALL = tuple(G.DEFAULT_RULES)
TICK = 0.02


def view(label: str = "walk", twist: float = 0.0, **kw: object) -> G.BodyView:
    return G.BodyView(label=label, twist=twist, **kw)  # type: ignore[arg-type]


def settled_guard(label: str = "walk", seconds: float = 4.0, **kw: object) -> G.SkillGuard:
    """A guard that has watched the body in one posture for ``seconds`` (from t = 0)."""
    g = G.SkillGuard(ALL, **kw)  # type: ignore[arg-type]
    for k in range(int(seconds / TICK) + 1):
        g.watch(view(label), k * TICK)
    return g


def test_postures_follow_robotd_labels_and_what_each_skill_was_gated_for() -> None:
    g = settled_guard()
    assert G.posture_of(view("walk", 0.0)) == "stand"
    assert G.posture_of(view("walk", 0.3)) == "walk"
    assert G.posture_of(view("startle_2:unwind"), g.names) == "skill"
    assert g.request("startle_2", view("walk", 0.4), 4.0).ok
    assert g.request("shake_1", view("walk", 0.4), 10.0).reason == "posture"
    assert g.request("stretch_sit", view("walk"), 10.0).reason == "posture"
    seated = settled_guard("sit", seconds=3.5)
    assert seated.request("stretch_sit", view("sit"), 3.5).ok
    assert seated.request("startle_1", view("sit"), 10.0).reason == "posture"  # a seated duck flips
    assert g.request("startle_1", view("rise"), 10.0).reason == "posture"


def test_upright_unheld_undocked_and_nothing_else_running() -> None:
    g = settled_guard()
    assert g.request("startle_1", view(tilt_deg=25.0), 4.0).reason == "tilt"
    assert g.request("startle_1", view(held=True), 4.0).reason == "held"
    assert g.request("startle_1", view(docked=True), 4.0).reason == "docked"
    assert g.request("startle_1", view(fallen=True), 4.0).reason == "fallen"
    assert g.request("startle_1", view("shake_1"), 4.0).reason == "busy"
    assert g.request("kick_left", view(), 4.0).reason == "unknown"  # (robotd's own, not in our rules)
    d = g.request("startle_1", view(held=True), 4.0)
    assert not d.ok and d.reason_zh == G.REASONS_ZH["held"]


def test_a_roll_is_asked_only_standing_and_settled_and_rarely() -> None:
    def asked(label: str = "stand", **kw: object) -> G.Decision:
        return settled_guard("stand", seconds=4.0).request("roulade", view(label, **kw), 4.0)

    assert asked("sit").reason == "posture"  # (sitting: it never rolls, it can tip)
    assert asked("walk", twist=0.3).reason == "posture"  # (a walk: stand first)
    assert asked(tilt_deg=25.0).reason == "tilt"
    assert asked(held=True).reason == "held"
    assert asked(docked=True).reason == "docked"
    assert asked(fallen=True).reason == "fallen"
    g = settled_guard("stand", seconds=4.0)
    assert g.request("roulade", view("stand"), 4.0).ok
    assert g.request("roulade", view("stand"), 5.0).reason == "gap"  # (120 s apart)
    spaced = settled_guard("stand", seconds=4.0)
    ok = 0
    for k in range(8):
        ok += spaced.request("roulade", view("stand"), 4.0 + 121.0 * k).ok
        spaced.watch(view("stand"), 4.0 + 121.0 * k + 1.0)
    assert ok == G.DEFAULT_RULES["roulade"].per_hour == 4  # (four an hour, however it is asked)
    fresh = G.SkillGuard(ALL)
    fresh.watch(view("stand"), 0.0)
    assert fresh.request("roulade", view("stand"), 0.5).reason == "settle"  # (stood 0.5 s: not 1.5)


def test_a_roll_is_never_handed_back_at_the_tilt_every_roll_passes() -> None:
    g = settled_guard("stand")
    assert g.watch(view("roulade", tilt_deg=120.0), 5.0) is None  # (the 30 degree watch would abort a bow here)
    assert g.watch(view("startle_2", tilt_deg=32.0), 6.0) == G.ABORT  # (the other skills keep it)


def test_a_hop_over_runs_through_its_own_lean_and_is_still_handed_back_falling() -> None:
    g = settled_guard()
    assert g.watch(view("hop_over_2", tilt_deg=38.0), 5.0) is None  # (a clean hop over the 20 mm sill leans to ~37-42)
    assert g.watch(view("hop_over_2:unwind", tilt_deg=44.0), 5.1) is None
    assert g.watch(view("hop_over_2", tilt_deg=60.0), 5.2) == G.ABORT  # (a fall goes on past it)
    h = settled_guard()
    assert h.watch(view("hop_over_4", tilt_deg=62.0), 5.0) is None  # (the 4 cm hop leans to ~57-66)
    assert h.watch(view("hop_over_4", tilt_deg=80.0), 5.1) == G.ABORT


def test_a_rebuilt_table_carries_the_hop_overs_own_lean() -> None:
    from duckmind.skilltrain import gt_wiring as W
    from duckmind.skilltrain import motion_table as MT

    doc = {"guard": MT.guard_block(["hop_over_2", "hop_over_4", "startle_2"])}
    g = G.SkillGuard(ALL, W.guard_rules(doc))
    for k in range(int(4.0 / TICK) + 1):
        g.watch(view(), k * TICK)
    assert g.watch(view("hop_over_2", tilt_deg=40.0), 5.0) is None
    assert g.watch(view("startle_2", tilt_deg=32.0), 6.0) == G.ABORT  # (the other skills keep the 30 degree watch)


def test_hourly_caps_gaps_and_the_global_cap() -> None:
    g = settled_guard()
    t = 4.0
    ok = 0
    for _ in range(20):
        if g.request("hop_back", view(), t).ok:
            ok += 1
        t += 61.0
    assert ok == G.DEFAULT_RULES["hop_back"].per_hour  # 4 in the hour
    assert g.request("hop_back", view(), t).reason == "hourly"
    assert g.request("hop_back", view(), t + 3600.0).ok  # the hour rolled on
    assert g.request("hop_back", view(), t + 3601.0).reason == "gap"
    small = settled_guard(global_per_hour=3)
    for k in range(3):
        assert small.request("startle_1", view(), 4.0 + k * 3.0).ok
    assert small.request("startle_2", view(), 20.0).reason == "total"
    assert small.counts(20.0)["all"] == 3


def test_release_and_abort_only_end_a_running_skill_and_abort_fires_once_per_run() -> None:
    g = G.SkillGuard(ALL)
    assert g.request(G.RELEASE, view("walk"), 0.0).reason == "nothing"
    assert g.request(G.RELEASE, view("freeze_4"), 0.0).ok
    assert g.request(G.RELEASE, view("shake_1"), 0.0).reason == "nothing"  # not a crouch
    assert g.watch(view("freeze_4", tilt_deg=10.0), 0.0) is None
    assert g.watch(view("freeze_4", tilt_deg=35.0), 0.1) == G.ABORT
    assert g.watch(view("freeze_4:unwind", tilt_deg=40.0), 0.2) is None  # once per run
    assert g.watch(view("walk", tilt_deg=40.0), 0.3) is None
    assert g.watch(view("startle_2", tilt_deg=31.0), 0.4) == G.ABORT


def test_comfort_skills_wait_for_the_settle_after_a_rise_a_stop_or_another_skill() -> None:
    g = G.SkillGuard(ALL)
    t = 0.0
    for _ in range(50):  # the seated-boot rise
        g.watch(view("rise"), t)
        t += TICK
    g.watch(view("walk"), t)  # velstand has taken over: robotd says walk, the twist is 0
    # the A100's "risen" entries (0.1–1.0 s after the hand-over): refused
    for dt in (0.2, 0.6):
        assert g.request("stretch_tall", view(), t + dt).reason == "settle"
        assert g.request("shake_2", view(), t + dt).reason == "settle"
        assert g.request("hop_back", view(), t + dt).reason == "settle"
    # a startle is a reflex: allowed 1.5 s after a rise (measured: 4/30 falls 0.1–1.0 s after)
    assert g.request("startle_4", view(), t + 0.7).reason == "settle"
    assert g.request("stretch_tall", view(), t + 1.0).reason == "settle"
    for k in range(1, 80):
        g.watch(view(), t + k * TICK)
    assert g.request("startle_2", view(), t + 1.6).ok
    for k in range(81, 90):
        g.watch(view(), t + k * TICK)
    assert g.request("stretch_tall", view(), t + 1.8).ok
    # a stop: walking, then standing (the twist gone) — the coast is not settled
    for k in range(100):
        g.watch(view("walk", 0.4), 10.0 + k * TICK)
    g.watch(view("walk", 0.0), 12.0)
    assert g.request("shake_1", view(), 12.3).reason == "settle"
    for k in range(1, 80):
        g.watch(view(), 12.0 + k * TICK)
    assert g.request("shake_1", view(), 13.6).ok
    # the trunk still rocking (gyro) keeps it waiting even after 1.5 s
    rocking = G.SkillGuard(ALL)
    for k in range(120):
        rocking.watch(view(gyro_rad_s=0.6 if k > 90 else 0.05), k * TICK)
    assert rocking.request("shake_1", view(gyro_rad_s=0.6), 2.4).reason == "settle"
    # a measured trunk speed (the coast) too
    assert settled_guard().request("shake_1", view(speed_mps=0.08), 4.0).reason == "settle"
    # seated: 3 s after sitting down (the glide), not before
    s = G.SkillGuard(ALL)
    for k in range(100):
        s.watch(view("sit"), k * TICK)
    assert s.request("stretch_sit", view("sit"), 2.0).reason == "settle"
    for k in range(100, 160):
        s.watch(view("sit"), k * TICK)
    assert s.request("stretch_sit", view("sit"), 3.2).ok


def test_walking_entries_only_as_fast_as_they_were_gated() -> None:
    rules = dict(G.DEFAULT_RULES)
    rules["freeze_4"] = G.Rule(**{**rules["freeze_4"].__dict__, "max_walk_mps": 0.40})
    g = G.SkillGuard(ALL, rules)
    for k in range(200):
        g.watch(view("walk", 0.55, vx=0.55), k * TICK)
    assert g.request("freeze_4", view("walk", 0.55, vx=0.55), 4.0).reason == "speed"
    assert g.request("startle_4", view("walk", 0.55, vx=0.55), 4.0).ok  # its verdict: ≤ 0.55
    assert g.request("freeze_4", view("walk", 0.41, vx=0.41), 60.0).ok  # within the tolerance
    # the measured trunk speed counts too (a walk faster than commanded)
    g2 = G.SkillGuard(ALL, rules)
    assert g2.request("freeze_4", view("walk", 0.40, vx=0.40, speed_mps=0.5), 0.0).reason == "speed"
    assert g2.request("hop_back", view("walk", 0.55, vx=0.55), 0.0).reason in ("speed", "settle")


def test_a_hold_is_refreshed_takes_over_a_startle_and_is_counted_once() -> None:
    g = settled_guard()
    assert g.request("startle_2", view(), 4.0).ok
    # the freeze takes over the running startle (in its hold or its unwind): no rise between
    d = g.request("freeze_3", view("startle_2:unwind"), 4.4)
    assert d.ok and d.reason == "continue"
    # re-requests of the running hold: refreshes, never refused, never counted
    for k in range(200):
        r = g.request("freeze_3", view("freeze_3"), 4.5 + k * TICK)
        assert r.ok and r.reason == "hold"
    assert g.counts(10.0)["per_skill"]["freeze_3"] == 1
    # a deeper hold may take over a shallower one; a one-shot may not take over anything
    assert g.request("freeze_4", view("freeze_3"), 9.0).reason == "continue"
    assert g.request("startle_4", view("freeze_4"), 9.1).reason == "busy"
    # a comfort skill running: no takeover
    assert g.request("freeze_4", view("shake_1"), 60.0).reason == "busy"


def test_strength_to_entry_the_fallback_and_the_picker() -> None:
    assert G.startle_entry(0.1) is None
    assert [G.startle_entry(s) for s in (0.2, 0.5, 0.7, 0.9)] == [
        "startle_1",
        "startle_2",
        "startle_3",
        "startle_4",
    ]
    assert G.startle_entry(0.7, lasting=True) == "freeze_3"
    assert G.startle_entry(0.9, lasting=True) == "freeze_4"
    g = settled_guard()
    for k in range(int(G.DEFAULT_RULES["startle_4"].per_hour or 0)):
        assert g.request("startle_4", view(), 4.0 + k * 11.0).ok
    d = G.request_with_fallback(g, "startle_4", view(), 200.0)
    assert d.ok and d.skill == "startle_3b"  # the next shallower level
    assert G.lower("startle_1b") == "startle_1" and G.lower("freeze_2b") == "freeze_2"
    assert G.lower("freeze_1b") is None
    # without a clock: the Stage-1 picker (one name per onset, the four levels)
    p = G.SkillPicker()
    stand: dict[str, Any] = {"posture": "stand", "bout": 0, "behaviour": "x"}
    assert p.pick(reflex="startle", intensity=0.5, **stand) == "startle_2"
    assert p.pick(reflex="startle", intensity=0.5, **stand) is None  # the same reflex: once
    assert p.pick(reflex=None, intensity=0.5, **stand) is None
    assert p.pick(reflex="startle", intensity=0.2, **stand) == "startle_1"  # a new onset
    assert (
        p.pick(reflex=None, behaviour="body_shake", bout=3, intensity=0.9, posture="stand")
        == "shake_2"
    )
    assert (
        p.pick(reflex=None, behaviour="stretch_yawn", bout=4, intensity=0.5, posture="sit")
        == "stretch_sit"
    )
    assert (
        p.pick(reflex=None, behaviour="freeze", bout=5, intensity=0.9, posture="walk") == "freeze_4"
    )
    assert p.pick(reflex=None, behaviour="flee", bout=6, intensity=0.9, posture="stand") is None
    assert (
        p.pick(
            reflex=None, behaviour="flee", bout=7, intensity=0.9, posture="stand", room_behind=True
        )
        == "hop_back"
    )
    assert (
        p.pick(
            reflex="startle",
            reflex_seq=9,
            behaviour="x",
            bout=7,
            intensity=1.0,
            posture="walk",
            room_behind=True,
        )
        == "hop_back"
    )
    assert (
        p.pick(reflex="startle", reflex_seq=10, behaviour="x", bout=7, intensity=1.0, posture="sit")
        is None
    )  # never from a seated duck


class Brain:
    """A scripted brain + robotd label for the clocked picker: the reflex / behaviour over
    time, and what robotd would be running given the picker's requests (holds keep running
    while re-requested; a one-shot runs 1.3 s; the release 0.8 s)."""

    def __init__(self, picker: G.SkillPicker) -> None:
        self.p = picker
        self.label = "walk"
        self.until = 0.0
        self.log: list[tuple[float, str]] = []

    def step(self, t: float, reflex: str | None, behaviour: str = "x", bout: int = 0,
             strength: float = 0.8) -> str | None:  # fmt: skip
        if self.label != "walk" and t >= self.until:
            self.label = "walk"
        posture = "skill" if self.label != "walk" else "stand"
        name = self.p.pick(reflex=reflex, behaviour=behaviour, bout=bout, intensity=0.5,
                           posture=posture, now_s=t, strength=strength,
                           label=self.label)  # fmt: skip
        if name is not None:
            base = self.label.split(":")[0]
            if name == G.RELEASE:
                self.label, self.until = G.RELEASE, t + 0.8
            elif name.startswith("freeze"):
                if base != name:
                    self.log.append((round(t, 2), name))
                self.label, self.until = name, t + 1.0
            elif self.label == "walk":
                self.log.append((round(t, 2), name))
                self.label, self.until = name, t + 1.3
            if name == G.RELEASE:
                self.log.append((round(t, 2), name))
        return name


def test_a_startle_the_brain_follows_with_a_freeze_is_one_crouch_and_one_rise() -> None:
    p = G.SkillPicker(seed=3)
    b = Brain(p)
    t = 0.0
    # the brain: startle (0.5 s) then its freeze (3.0 s), then back to its act
    while t < 8.0:
        reflex = "startle" if 1.0 <= t < 1.5 else "freeze" if 1.5 <= t < 4.5 else None
        b.step(t, reflex, strength=0.8)
        t += TICK
    kinds = [n for _, n in b.log]
    assert kinds[0].startswith("freeze") and kinds[-1] == G.RELEASE
    assert sum(1 for n in kinds if n == G.RELEASE) == 1  # one rise
    assert not any(n.startswith("startle_") and n != G.RELEASE for n in kinds)  # no one-shot
    released = [t for t, n in b.log if n == G.RELEASE][0]
    # held at least its drawn minimum, at most through the brain's freeze (+ the tail) — its own
    # drawn length may end it earlier; the brain's freeze then goes on as stillness, upright
    assert 1.0 + G.HOLD_MIN_S - 0.05 <= released <= 4.5 + G.HOLD_TAIL_S + 0.05


def test_a_freeze_takes_over_a_running_one_shot_and_none_follows_a_finished_one() -> None:
    p = G.SkillPicker(seed=1)
    b = Brain(p)
    t = 0.0
    while t < 3.0:  # a weak startle (a one-shot) and, 0.7 s later, the brain's freeze bout
        reflex = "startle" if 1.0 <= t < 1.4 else None
        beh, bout = ("freeze", 2) if t >= 1.7 else ("x", 1)
        b.step(t, reflex, beh, bout, strength=0.18)
        t += TICK
    names = [n for _, n in b.log]
    assert names[0].startswith("startle_") and names[1].startswith("freeze")
    one_level = G.LEVEL[names[0]]
    assert G.LEVEL[names[1]] >= one_level  # never shallower: no rise before the freeze
    # a freeze bout arriving just after a one-shot finished: no second crouch
    q = G.SkillPicker(seed=2)
    c = Brain(q)
    t = 0.0
    while t < 5.0:
        reflex = "startle" if 1.0 <= t < 1.2 else None
        beh, bout = ("freeze", 2) if t >= 2.6 else ("x", 1)
        c.step(t, reflex, beh, bout, strength=0.18)
        t += TICK
    assert [n for _, n in c.log if n.startswith("freeze")] == []


def test_every_execution_draws_its_own_depth_and_hold() -> None:
    depths: list[float] = []
    holds: list[float] = []
    for seed in range(300):
        p = G.SkillPicker(seed=seed)
        b = Brain(p)
        t = 0.0
        start = end = None
        while t < 14.0:
            name = b.step(t, "startle" if 1.0 <= t < 1.1 else None, strength=0.7)
            if name and name.startswith("freeze") and start is None:
                start = t
                depths.append(G.LEVEL[name])
            if name == G.RELEASE:
                end = t
                break
            t += TICK
        assert start is not None and end is not None
        holds.append(end - start)
    cv = statistics.pstdev(depths) / statistics.mean(depths)
    assert 0.08 <= cv <= 0.3, cv  # CV 0.1–0.3 (the grid quantises a little)
    assert len(set(depths)) >= 3
    assert min(holds) >= G.HOLD_MIN_S - 0.05 and max(holds) <= G.HOLD_MAX_S + 0.05
    hcv = statistics.pstdev(holds) / statistics.mean(holds)
    assert 0.1 <= hcv <= 0.4, hcv


def test_a_refused_hold_falls_back_to_a_one_shot_and_the_picker_only_names_loaded_entries() -> None:
    p = G.SkillPicker(seed=5, available=frozenset({"startle_2", "startle_4"}))
    # no hold loaded: a lasting startle is a one-shot at the nearest loaded level
    name = p.pick(reflex="startle", behaviour="x", bout=0, intensity=0.5, posture="stand",
                  now_s=1.0, strength=0.55, label="walk")  # fmt: skip
    assert name in ("startle_2", "startle_4")
    q = G.SkillPicker(seed=5)
    first = q.pick(reflex="startle", behaviour="x", bout=0, intensity=0.5, posture="stand",
                   now_s=1.0, strength=0.9, label="walk")  # fmt: skip
    assert first is not None and first.startswith("freeze")
    # robotd never ran it (the guard refused: capped): after CONFIRM_S the one-shot instead
    later: list[str] = []
    for k in range(1, 30):
        name = q.pick(reflex="startle", behaviour="x", bout=0, intensity=0.5, posture="stand",
                      now_s=1.0 + k * TICK, strength=0.9, label="walk")  # fmt: skip
        if name is not None:
            later.append(name)
    # re-requested while robotd may still be starting it, then the one-shot, then nothing
    assert later[-1].startswith("startle_") and later.count(later[-1]) == 1
    assert all(n == first for n in later[:-1]) and len(later) - 1 <= G.CONFIRM_S / TICK + 1


def test_caps_from_the_strain_gates() -> None:
    rules = G.caps_from_strain(
        G.DEFAULT_RULES,
        {
            "shake_2": {"tau_leg_p95": 0.40, "tau_head_p95": 0.2, "stall_frac": 0.0},
            "shake_1": {"tau_leg_p95": 0.32, "tau_head_p95": 0.2, "stall_frac": 0.0},
            "startle_1": {"tau_leg_p95": 0.2, "tau_head_p95": 0.1, "stall_frac": 0.0},
            # a condition summary as metrics.summarize writes it
            "hop_back": {
                "strain": {"tau_leg_p95": 0.2, "tau_head_p95": 0.1, "stall_frac": 0.0},
                "servo_hits_upright": {"k": 1, "n": 30},
            },
        },
    )
    assert rules["shake_2"].per_hour == 0
    assert rules["shake_1"].per_hour == (G.DEFAULT_RULES["shake_1"].per_hour or 0) // 2
    assert rules["startle_1"].per_hour == G.DEFAULT_RULES["startle_1"].per_hour
    assert rules["hop_back"].per_hour == 0  # a housing hit while upright


def test_a_startle_starts_from_the_brains_event_not_its_later_onset() -> None:
    p = G.SkillPicker(seed=4)
    kw: dict[str, Any] = {"behaviour": "x", "bout": 0, "intensity": 0.5, "posture": "stand"}
    # t = 1.00: the brain perceives the clap (its event); its reflex name follows at 1.06
    first = p.pick(reflex=None, now_s=1.00, label="walk", event=(1000, "startle", 0.9), **kw)
    assert first is not None and first.startswith("freeze")  # a strong startle: held
    for k in range(1, 6):
        name = p.pick(reflex="startle", now_s=1.0 + 0.02 * k, label=first, strength=0.9,
                      event=(1000, "startle", 0.9), **kw)  # fmt: skip
        assert name == first  # the hold goes on — no second start at the reflex's onset
    q = G.SkillPicker(seed=4)
    weak = q.pick(reflex=None, now_s=1.0, label="walk", event=(1000, "startle", 0.2), **kw)
    assert weak is not None and weak.startswith("startle_")
    again = q.pick(reflex="startle", now_s=1.06, label=weak, strength=0.2,
                   event=(1000, "startle", 0.2), **kw)  # fmt: skip
    assert again is None  # the same startle


def test_no_takeover_once_the_one_shot_unwinds() -> None:
    p = G.SkillPicker(seed=9)
    kw: dict[str, Any] = {"intensity": 0.9, "posture": "skill", "now_s": 2.0, "strength": 0.9}
    # the freeze arrives while startle_2 is already rising (its unwind): no second crouch
    assert p.pick(reflex="freeze", behaviour="x", bout=1, label="startle_2:unwind", **kw) is None
    # … while it still holds its crouch: taken over, at least as deep
    q = G.SkillPicker(seed=9)
    name = q.pick(reflex="freeze", behaviour="x", bout=1, label="startle_2", **kw)
    assert name is not None and name.startswith("freeze") and G.LEVEL[name] >= 0.5


def test_the_crouch_follows_the_brains_freeze_reflex_not_the_freeze_bout_after_it() -> None:
    p = G.SkillPicker(seed=11)
    b = Brain(p)
    t = 0.0
    released = None
    while t < 14.0:
        # the brain: startle 0.5 s, its freeze reflex 2.5 s, then a freeze BOUT for 8 s more
        reflex = "startle" if 1.0 <= t < 1.5 else "freeze" if 1.5 <= t < 4.0 else None
        beh, bout = ("freeze", 3) if 1.2 <= t < 12.0 else ("x", 1)
        name = b.step(t, reflex, beh, bout, strength=0.8)
        if name == G.RELEASE:
            released = t
            break
        t += TICK
    assert released is not None and released <= 4.0 + G.HOLD_TAIL_S + 3.0  # not held to the cap
    # the bout that was already running when it rose starts no second crouch
    for k in range(1, 200):
        name = b.step(released + k * TICK, None, "freeze", 3, strength=0.8)
        assert name is None or name == G.RELEASE
