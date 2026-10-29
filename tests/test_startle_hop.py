"""A startle includes two small steps back (2026-10-10, docs/design/neural-duck/STARTLE-HOP-2026-10-10.md)."""

from __future__ import annotations

import math

from duckmind.duckbody.robotd_client import RobotState
from duckmind.neuro import motions as MO
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.skilltrain import guard as G

ALL = tuple(G.DEFAULT_RULES)
TICK = 0.02


def view(label: str = "walk", twist: float = 0.0, **kw: object) -> G.BodyView:
    return G.BodyView(label=label, twist=twist, **kw)  # type: ignore[arg-type]


def guard_after(label: str = "walk", seconds: float = 0.4) -> G.SkillGuard:
    g = G.SkillGuard(ALL)
    for k in range(int(seconds / TICK) + 1):
        g.watch(view(label), k * TICK)
    return g


# ── the guard ──


def test_a_chained_hop_waits_for_the_stop_not_for_the_settle() -> None:
    g = guard_after(seconds=0.4)  # standing for 0.4 s only: a plain hop-back says settle
    assert g.request("hop_back", view(), 0.5).reason == "settle"
    slow = view(speed_mps=0.02, tilt_deg=2.0)
    assert g.request("hop_back", slow, 0.5, chain=True).ok
    g2 = guard_after(seconds=0.4)
    assert (
        g2.request("hop_back", view(speed_mps=0.2, tilt_deg=2.0), 0.5, chain=True).reason
        == "settle"
    )
    assert (
        g2.request("hop_back", view(speed_mps=0.04, tilt_deg=9.0), 0.5, chain=True).reason
        == "settle"
    )
    assert (
        g2.request("hop_back", view(tilt_deg=2.0), 0.5, chain=True).reason == "settle"
    )  # speed unknown
    # the safety rules are the same as for any run
    assert g2.request("hop_back", view(speed_mps=0.0, held=True), 0.5, chain=True).reason == "held"
    assert (
        g2.request("hop_back", view(speed_mps=0.0, righting=True), 0.5, chain=True).reason
        == "fallen"
    )
    stand_only = dict(G.DEFAULT_RULES)  # (the installed table: the hop-back from standing only)
    stand_only["hop_back"] = G.replace(stand_only["hop_back"], postures=frozenset({"stand"}))
    g3 = G.SkillGuard(ALL, stand_only)
    for k in range(21):
        g3.watch(view(), k * TICK)
    assert (
        g3.request("hop_back", view("walk", 0.3, speed_mps=0.0), 0.5, chain=True).reason
        == "posture"
    )
    assert g2.request("hop_back", view("sit", speed_mps=0.0), 0.5, chain=True).reason == "posture"


def test_no_hop_within_a_second_and_a_half_of_a_rise_scare_or_not() -> None:
    g = G.SkillGuard(ALL)
    g.watch(view("rise"), 0.0)
    g.watch(view("walk"), 0.5)
    assert (
        g.request("hop_back", view(speed_mps=0.0, tilt_deg=1.0), 0.6, chain=True).reason == "settle"
    )
    assert g.request("hop_back", view(speed_mps=0.0, tilt_deg=1.0), 1.6, chain=True).ok


def test_a_hop_that_follows_a_scare_counts_half_against_the_hourly_cap() -> None:
    rules = dict(G.DEFAULT_RULES)
    rules["hop_back"] = G.replace(rules["hop_back"], per_hour=2, min_gap_s=60.0)
    g = G.SkillGuard(ALL, rules)
    g.watch(view(), 0.0)
    ok, t = 0, 1.0
    for _ in range(10):
        g.watch(view(), t - 0.5)
        if g.request("hop_back", view(speed_mps=0.0, tilt_deg=1.0), t, chain=True).ok:
            ok += 1
        t += 61.0
    assert ok == 4  # a cap of 2 an hour: 4 of the half-weight runs
    c = g.counts(t)
    assert c["per_skill"]["hop_back"] == 4 and c["chained"]["hop_back"] == 4
    # a plain run counts full: 1 plain + 2 chained fill a cap of 2
    h2 = G.SkillGuard(ALL, rules)
    for k in range(int(3.0 / TICK) + 1):
        h2.watch(view(), k * TICK)
    assert h2.request("hop_back", view(), 3.0).ok  # plain: 1
    assert h2.request("hop_back", view(speed_mps=0.0, tilt_deg=1.0), 64.0, chain=True).ok  # +0.5
    assert h2.request(
        "hop_back", view(speed_mps=0.0, tilt_deg=1.0), 125.0, chain=True
    ).ok  # +0.5 = 2
    assert (
        h2.request("hop_back", view(speed_mps=0.0, tilt_deg=1.0), 186.0, chain=True).reason
        == "hourly"
    )
    # a person's press is never counted
    assert h2.request("hop_back", view(), 187.0, manual=True).ok


# ── the picker ──


class Robotd:
    """A scripted body for the clocked picker: robotd's label over time (a skill runs for the lengths of the real table) and
    what the brain feeds the picker (hop_ok, still, the last answer to the hop ask)."""

    DURATION = {"hop_back": 1.4}

    def __init__(
        self, picker: G.SkillPicker, *, still_after: float = 0.0, hop_ok: bool | None = True
    ) -> None:
        self.p = picker
        self.label = "walk"
        self.twist = 0.0
        self.until = 0.0
        self.log: list[tuple[float, str, bool]] = []
        self.still_at = still_after
        self.hop_ok = hop_ok
        self.decision: tuple[bool, str, float] | None = None
        self.refuse: str | None = None

    def step(self, t: float, reflex: str | None, behaviour: str = "x", bout: int = 0, strength: float = 0.9,
             walking: bool = False) -> str | None:  # fmt: skip
        if self.label != "walk" and t >= self.until:
            self.label = "walk"
        posture = "skill" if self.label != "walk" else ("walk" if walking else "stand")
        name = self.p.pick(reflex=reflex, behaviour=behaviour, bout=bout, intensity=0.5, posture=posture,
                           now_s=t, strength=strength, label=self.label, hop_ok=self.hop_ok,
                           still=(t >= self.still_at), hop_decision=self.decision)  # fmt: skip
        if name is not None:
            if name == "hop_back":
                if self.refuse:
                    self.decision = (False, self.refuse, t)
                elif self.label == "walk":
                    self.decision = (True, "ok", t)
                    self.label, self.until = "hop_back", t + self.DURATION["hop_back"]
                    self.log.append((round(t, 2), name, self.p.chain))
            elif name.startswith("freeze"):  # (a hold keeps running while it is re-requested)
                if self.label.split(":")[0] != name:
                    self.log.append((round(t, 2), name, self.p.chain))
                self.label, self.until = name, t + 1.0
            elif self.label == "walk" and name != G.RELEASE:
                self.log.append((round(t, 2), name, self.p.chain))
                self.label, self.until = name, t + 1.3
            elif name == G.RELEASE:
                self.label, self.until = G.RELEASE, t + 0.8
                self.log.append((round(t, 2), name, False))
        return name


def run(
    b: Robotd,
    seconds: float,
    *,
    scare_at: float = 1.0,
    walking_until: float = 0.0,
    strength: float = 0.9,
) -> None:
    t = 0.0
    while t < seconds:
        reflex = (
            "startle"
            if scare_at <= t < scare_at + 0.4
            else "freeze"
            if scare_at + 0.4 <= t < scare_at + 3.0
            else None
        )
        b.step(t, reflex, strength=strength, walking=t < walking_until)
        t += TICK


def test_a_standing_duck_answers_a_scare_of_level_two_or_more_with_the_back_steps() -> None:
    for seed in range(8):
        p = G.SkillPicker(seed=seed)
        b = Robotd(p)
        run(b, 6.0, strength=0.9)
        assert [n for _, n, _ in b.log] == ["hop_back"], (
            seed,
            b.log,
        )  # the steps ARE the startle: no dip, no crouch after it
        assert b.log[0][2] is True  # asked as the chained hop
        assert b.log[0][0] <= 1.1  # at the scare, not later
        assert p.hop_t is not None and p.follow is None


def test_a_small_scare_is_the_plain_dip_and_a_blocked_way_or_a_scare_from_behind_is_todays_startle() -> (
    None
):
    p = G.SkillPicker(seed=1)
    b = Robotd(p)
    run(b, 5.0, strength=0.2)
    assert [n for _, n, _ in b.log][0].startswith("startle_1") and "hop_back" not in [
        n for _, n, _ in b.log
    ]
    q = G.SkillPicker(seed=1)
    c = Robotd(
        q, hop_ok=False
    )  # the caller says the way back is blocked / the scare came from behind
    run(c, 14.0, strength=0.9)
    names = [n for _, n, _ in c.log]
    assert names[0].startswith("freeze") and G.RELEASE in names and "hop_back" not in names


def test_without_hop_ok_the_old_rule_holds() -> None:
    p = G.SkillPicker(seed=1)
    pick = p.pick(reflex="startle", reflex_seq=1, behaviour="x", bout=0, intensity=1.0, posture="stand",
                  now_s=1.0, strength=1.0, label="walk", room_behind=False)  # fmt: skip
    assert pick is not None and pick.startswith("freeze")  # a held crouch, as before


def test_a_scare_that_finds_the_duck_walking_waits_for_the_stop_and_then_steps_back() -> None:
    p = G.SkillPicker(seed=2)
    b = Robotd(p, still_after=1.6)
    run(b, 6.0, scare_at=1.0, walking_until=1.5, strength=0.9)
    assert [n for _, n, _ in b.log] == ["hop_back"], b.log
    assert 1.5 <= b.log[0][0] <= 2.2  # right after it stood still


def test_a_walking_scare_that_does_not_stop_in_a_second_gets_the_plain_startle() -> None:
    p = G.SkillPicker(seed=2)
    b = Robotd(p, still_after=99.0)  # never slow
    run(b, 6.0, scare_at=1.0, walking_until=0.0, strength=0.9)
    names = [n for _, n, _ in b.log]
    assert "hop_back" not in names
    assert names and names[0].startswith("freeze")
    assert 2.0 <= b.log[0][0] <= 2.2  # the wait was HOP_WAIT_S, no more


def test_a_hard_refusal_of_the_hop_turns_into_the_plain_startle_at_once() -> None:
    p = G.SkillPicker(seed=3)
    b = Robotd(p)
    b.refuse = "gap"
    run(b, 6.0, scare_at=1.0, strength=0.9)
    names = [n for _, n, _ in b.log]
    assert "hop_back" not in names and names[0].startswith("freeze")
    assert b.log[0][0] < 1.3
    # a soft one (the trunk not slow enough yet) is waited out
    q = G.SkillPicker(seed=3)
    c = Robotd(q)
    c.refuse = "settle"
    t = 0.0
    while t < 1.5:
        c.step(t, "startle" if 1.0 <= t < 1.2 else None)
        t += TICK
    assert q.follow is not None
    c.refuse = None
    while t < 2.5:
        c.step(t, None)
        t += TICK
    assert [n for _, n, _ in c.log] == ["hop_back"]


def test_the_flight_is_not_a_second_hop() -> None:
    p = G.SkillPicker(seed=4)
    pick = p.pick(reflex=None, behaviour="flee", bout=7, intensity=0.9, posture="stand", room_behind=True,
                  now_s=3.0, strength=0.9, label="walk", hop_ok=True, still=True)  # fmt: skip
    assert pick is None
    legacy = G.SkillPicker(seed=4).pick(reflex=None, behaviour="flee", bout=7, intensity=0.9, posture="stand",
                                        room_behind=True)  # fmt: skip
    assert legacy == "hop_back"


def test_with_a_dip_first_the_steps_come_after_the_dip_has_run() -> None:
    G_dip = G.HOP_DIP_FIRST
    G.HOP_DIP_FIRST = True
    try:
        p = G.SkillPicker(seed=5)
        b = Robotd(p)
        run(b, 8.0, scare_at=1.0, strength=0.9)
        names = [n for _, n, _ in b.log]
        assert names[0].startswith("startle_") and names[1] == "hop_back", b.log
        assert b.log[1][0] >= b.log[0][0] + 1.3 - 0.05  # after the dip's 1.3 s
    finally:
        G.HOP_DIP_FIRST = G_dip


def test_a_press_of_a_startle_button_runs_the_dip_and_then_the_steps() -> None:
    p = G.SkillPicker(seed=6)
    b = Robotd(p)
    p.play("startle_3", 1.0, then="hop_back")
    t, seen_hold = 1.0, False
    while t < 8.0:
        b.step(t, None)
        seen_hold = seen_hold or p.hold_still
        t += TICK
    names = [n for _, n, _ in b.log]
    assert names == ["startle_3", "hop_back"], b.log
    assert b.log[1][0] >= 1.0 + 1.3 - 0.05
    assert seen_hold  # the walk is kept stopped while it waits
    assert p.follow is None


# ── the body link ──


def state(t: float, *, policy: str = "walk", tilt: float = 2.0, x: float = 0.0) -> RobotState:
    joints = [0.0] * 15
    for i, home in enumerate((0.3491, 0.3491, 0.0, 0.0)):
        joints[5 + i] = home
    r = math.radians(tilt)
    return RobotState.from_wire(
        {
            "t_ns": 10**10 + int(round(t * 1e9)),
            "policy": policy,
            "move": {"requested": [0, 0, 0], "applied": [0, 0, 0]},
            "safety": {"fallen": False, "limp": False, "gravity": [math.sin(r), 0.0, -math.cos(r)]},
            "joints": joints,
            "currents_ma": [4.5] * 15,
            "imu": {"gyro": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            "odom": {"position": [x, 0.0, 0.0], "yaw": 0.0},
        }
    )


def dos(out) -> list[str]:
    return [i.params.get("skill") for i in out.intents if i.method == "robot.do"]


def test_the_body_link_starts_a_chained_hop_once_the_trunk_is_slow() -> None:
    link = BodyLink()
    link.skills = G.SkillGuard(G.DEFAULT_RULES)
    for k in range(15):  # 0.3 s of standing: far less than the 1.5 s settle
        link.tick(state(k * 0.02), BodyRequest())
    plain = link.tick(state(0.32), BodyRequest(skill="hop_back"))
    assert dos(plain) == [] and "skill" in plain.gate
    out = link.tick(state(0.34), BodyRequest(skill="hop_back", skill_chain=True))
    assert dos(out) == ["hop_back"]
    assert link.skills.counts(0.4)["chained"] == {"hop_back": 1}
    # a trunk still moving (the odometry runs 0.5 m/s) waits
    moving = BodyLink()
    moving.skills = G.SkillGuard(G.DEFAULT_RULES)
    for k in range(15):
        moving.tick(state(k * 0.02, x=0.01 * k), BodyRequest())
    out = moving.tick(state(0.32, x=0.16), BodyRequest(skill="hop_back", skill_chain=True))
    assert dos(out) == []
    assert (
        moving.last_skill_decision is not None and moving.last_skill_decision[1].reason == "settle"
    )


def test_a_low_battery_makes_no_back_steps() -> None:
    link = BodyLink()
    link.skills = G.SkillGuard(G.DEFAULT_RULES)
    for k in range(15):
        link.tick(state(k * 0.02), BodyRequest(), battery_pct=10)
    out = link.tick(state(0.32), BodyRequest(skill="hop_back", skill_chain=True), battery_pct=10)
    assert "hop_back" not in dos(out)


# ── the page ──


def test_the_startle_buttons_of_level_two_to_four_add_the_steps() -> None:
    assert MO.BY_ID["startle_1"].then == ""
    for n in (2, 3, 4):
        m = MO.BY_ID[f"startle_{n}"]
        assert m.then == "hop_back" and "hop_back" in m.needs() and f"startle_{n}" in m.needs()
