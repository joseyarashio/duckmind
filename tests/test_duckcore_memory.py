from datetime import timedelta

import pytest

from duckmind.duckcore import query
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.duckcore.memory import DIARY_BANNED


@pytest.fixture()
def raised() -> Session:
    session = Session(Config(seed=7))
    dispatch(session, "live", {"days": 20, "routine": "daily"})
    return session


def test_forget_removes_the_fact_and_rolls_back_what_it_taught(raised):
    core = raised.core
    favourite = core.memory().facts["F-fav-verb"].value
    key = f"household.verb:{favourite}"
    prefs_before = dict(core.s.prefs)
    assert prefs_before[key] > 5000
    result = dispatch(raised, "forget", {"fact_id": "F-fav-verb"})
    assert result["ok"] and not result["still_there"]
    assert result["rolled_back"] > 0
    assert core.s.prefs[key] == 5000  # taught by nothing else, so back to neutral
    remaining = core.memory().facts.get("F-fav-verb")
    assert remaining is None or remaining.value != favourite  # another favourite may take its place
    assert core.log.verify()

    dispatch(raised, "undo_forget", {})
    assert core.s.prefs == prefs_before
    assert core.memory().facts["F-fav-verb"].value == favourite


def test_forgetting_can_only_be_undone_within_a_day(raised):
    dispatch(raised, "forget", {"fact_id": "F-touch-crown"})
    raised.core.advance(25 * 60)
    assert "超過 24 小時" in dispatch(raised, "undo_forget", {})["note"]
    assert "F-touch-crown" not in raised.core.memory().facts


def test_growth_and_personality_cannot_be_forgotten_piecemeal(raised):
    facts = raised.core.memory().facts
    for fact_id in ("F-ms-hatched", "F-quirk-tilt_side"):
        if fact_id in facts:
            assert dispatch(raised, "forget", {"fact_id": fact_id})["ok"] is False


def test_every_fact_cites_real_events(raised):
    core = raised.core
    for fact in core.memory().facts.values():
        assert fact.evidence, fact.fact_id
        assert all(core.log.get(event_id) is not None for event_id in fact.evidence), fact.fact_id


def test_the_diary_never_guilt_trips_even_when_neglected():
    session = Session(Config(seed=4))
    dispatch(session, "live", {"days": 60, "routine": "neglect"})
    text = " ".join(line["text"] for day in query.diary(session.core, 60) for line in day["lines"])
    assert text
    assert not [word for word in DIARY_BANNED if word in text]


def test_the_duck_admits_it_cannot_tell_people_apart(raised):
    answer = query.search(raised.core, "小孩喜歡什麼")["answer"]
    assert "分不出" in answer


def test_voice_stays_as_daily_counts_only(raised):
    core = raised.core
    core.power_on("owner")
    for _ in range(5):
        core.say("child")
    core.power_off("owner")
    core.advance(24 * 60)
    kinds = {event.kind for event in core.log.events}
    assert "env.daily_sound" in kinds
    assert not [kind for kind in kinds if kind.startswith(("sense.voice", "sense.audio"))]
    stamped = [event for event in core.log.events if event.kind == "env.daily_sound"]
    assert set(stamped[-1].payload) == {"date", "voice", "noise", "tv_minutes"}
    assert core.s.now - core.s.hatched_at > timedelta(days=20)


# ------------------------------------------------------------------ phase 4: new record kinds
#
# The same life written the way a v4 Duck Core wrote it and the way a v4_1 Duck Core with
# LegacyWorld writes it (design §8; docs/design/phase4-boundary.md §3-§4) must distill to the
# same memory: the same facts, episodes and diary, also after forgetting the falls.

LEGACY = ("legacy", "legacy", 1.0)
SELF = ("self", "duckcore", 1.0)
IMU = ("imu", "imu", 0.9)


class Life:
    """Writes one scripted life into an EventLog, in v4 or in v4_1 record shapes."""

    def __init__(self, v4_1: bool) -> None:
        from datetime import datetime, timezone

        from duckmind.duckcore.memory import EventLog

        self.v4_1 = v4_1
        self.log = EventLog()
        self.now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone(timedelta(hours=8)))
        self.acts = 0
        self.walk_events: list = []
        self.falls: list = []

    def emit(self, kind, payload, source=SELF, parents=()):
        return self.log.append(kind, self.now, source, payload, "life", tuple(parents))

    def percept(self, kind, parents=(), **fields):
        stamp = {"t_ms": int(self.now.timestamp()) * 1000, "pv": "l:0123456789abcdef"}
        return self.emit(kind, {**fields, **stamp}, LEGACY, parents)

    def act(self, cmd, **fields):
        self.acts += 1
        act_id = f"A{self.acts}"
        payload = {"cmd": cmd, **fields, "act_id": act_id, "deadline_ms": 0}
        return act_id, self.emit("duck.act", payload)

    def stand(self) -> None:
        if self.v4_1:
            act_id, command = self.act("duck.posture", want="stand")
            self.percept(
                "body.posture", (command.event_id,), want="stand", got="stand", ms=0,
                fell=False, act_id=act_id,
            )  # fmt: skip
        else:
            self.emit(
                "duck.posture",
                {"posture": "stand", "from": "sit", "reason": "restless", "by": "self"},
            )

    def walk(self, cm: int, seconds: int, fall_after: int | None = None) -> None:
        payload = {"verb": "wander", "seconds": seconds, "cm": cm, "meters": cm / 100}
        act_id = None
        if self.v4_1:
            self.acts += 1
            act_id = f"A{self.acts}"
            payload |= {"act_id": act_id, "max_ms": seconds * 1000, "vx_mm": 200, "vyaw_mrad": 0}
        walk = self.emit("duck.walk", payload)
        self.walk_events.append(walk)
        if fall_after is None:
            self.now += timedelta(seconds=seconds)
            if self.v4_1:
                self.percept(
                    "walk.done", (walk.event_id,), act_id=act_id, reason="time", steps=seconds * 4,
                    cmd={"vx": 200, "vyaw": 0, "max_ms": seconds * 1000},
                    odom={"dx_mm": cm * 10, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": cm * 10},
                )  # fmt: skip
            return
        self.now += timedelta(seconds=fall_after)
        short = cm - cm * fall_after // seconds
        cut = {
            "walk": walk.event_id,
            "after_s": fall_after,
            "cm_short": short,
            "pose": [1.9, 1.9, 0],
        }
        if self.v4_1:
            fell = self.percept(
                "body.fell", (walk.event_id,), during="walk", ctx={"act_id": act_id}
            )
            self.emit(
                "duck.posture",
                {"posture": "fallen", "from": "walk", "reason": "fell", "by": "self",
                 "during": "wander", "after_s": fall_after, "falls": len(self.falls) + 1,
                 "walk_cut": cut, "at": [1.9, 1.9, 0]},
                parents=(fell.event_id,),
            )  # fmt: skip
            self.percept(
                "walk.done", (walk.event_id,), act_id=act_id, reason="fell", steps=fall_after * 4,
                cmd={"vx": 200, "vyaw": 0, "max_ms": seconds * 1000},
                odom={"dx_mm": 0, "dy_mm": 0, "dyaw_mrad": 0, "path_mm": 0},
            )  # fmt: skip
        else:
            fell = self.emit(
                "body.fell",
                {"during": "wander", "after_s": fall_after, "falls": len(self.falls) + 1,
                 "walk_cut": cut, "at": [1.9, 1.9, 0]},
                IMU, (walk.event_id,),
            )  # fmt: skip
        self.falls.append(fell)

    def righting(self, oks: list[bool]) -> None:
        """Self-righting tries after the last fall (v4: its outcome on the last try)."""
        fell = self.falls[-1]
        for index, ok in enumerate(oks, 1):
            self.now += timedelta(seconds=2 if index == 1 else 4)
            last = ok or index == 3
            if self.v4_1:
                self.percept("body.righting", (fell.event_id,), ok=ok, ms=0, tries=index)
            else:
                payload = {"try": index, "of": 3, "ok": ok, "at": [1.9, 1.9, 0]}
                if last:
                    payload["outcome"] = ("easy" if index == 1 else "hard") if ok else "stuck"
                self.emit("body.righting", payload, IMU, (fell.event_id,))
            if ok:
                self.emit(
                    "duck.posture",
                    {"posture": "sit", "from": "fallen", "reason": "righted", "by": "self"},
                )

    def helped(self, tries: int) -> None:
        """A hand set it upright while it was still trying."""
        self.now += timedelta(seconds=1)
        outcome = "hard" if tries else "helped"
        if self.v4_1:
            self.emit(
                "duck.righting_end", {"helped": "set_upright", "outcome": outcome, "tries": tries}
            )
        else:
            self.emit(
                "body.righting",
                {"try": tries, "of": 3, "ok": False, "helped": "set_upright", "outcome": outcome},
                IMU,
                (self.falls[-1].event_id,),
            )
        self.emit(
            "duck.posture",
            {"posture": "sit", "from": "fallen", "reason": "set_upright", "by": "body"},
        )

    def stale_results(self) -> None:
        """Late answers no open act wanted (v4_1 only): recorded, never used."""
        if self.v4_1:
            self.percept(
                "body.posture",
                want="stand",
                got="stand",
                ms=0,
                fell=False,
                act_id="A99",
                stale=True,
            )
            self.percept("body.righting", ok=True, ms=0, tries=1, stale=True)
            self.percept("body.fell", during="walk", ctx={"act_id": "A98"}, stale=True)


def live_a_life(v4_1: bool, *, migrated_on_day: int | None = None) -> Life:
    life = Life(v4_1)
    for day in range(3):
        if migrated_on_day is not None:
            life.v4_1 = day >= migrated_on_day
        life.now = life.now.replace(day=1 + day, hour=9, minute=0)
        life.stand()
        life.walk(60, 5)
        life.walk(100, 5, fall_after=3)  # hard: twice
        life.righting([False, True])
        life.now += timedelta(minutes=10)
        life.stand()
        life.walk(80, 4)
        life.stale_results()
        life.walk(120, 6, fall_after=2)  # easy
        life.righting([True])
        life.now += timedelta(minutes=10)
        life.stand()
        life.walk(90, 3, fall_after=1)  # stuck, then a hand
        life.righting([False, False, False])
        life.now += timedelta(minutes=5)
        life.emit(  # its tries are over: a hand sets it upright later
            "duck.posture",
            {"posture": "sit", "from": "fallen", "reason": "set_upright", "by": "body"},
        )
        life.now += timedelta(minutes=10)
        life.stand()
        life.walk(50, 5, fall_after=4)  # a hand before its first try
        life.helped(0)
        life.now += timedelta(minutes=10)
        life.stand()
        life.walk(40, 2)
    return life


def memory_view(life: Life, forget_falls: bool = False):
    from duckmind.duckcore.memory import distill

    if forget_falls:
        life.emit(
            "memory.forget",
            {"event_ids": [fell.event_id for fell in life.falls]},
            ("owner", "app", 1.0),
        )
    memory = distill(life.log.events, life.now + timedelta(days=1), "小鴨")
    facts = {
        fact_id: (fact.category, fact.text, fact.value, fact.strength, len(fact.evidence))
        for fact_id, fact in memory.facts.items()
    }
    episodes = [(e.kind, e.start, e.end, e.summary, e.salience) for e in memory.episodes]
    diary = [
        (d.date, d.positive, d.negative, [line.text for line in d.lines]) for d in memory.diary
    ]
    return facts, episodes, diary


def test_v4_1_records_distill_to_the_v4_memory():
    old, new = live_a_life(False), live_a_life(True)
    facts, episodes, diary = memory_view(new)
    assert (facts, episodes, diary) == memory_view(old)
    assert facts["F-fell"][1].startswith("我走路時跌倒過 12 次，其中 3 次自己一下就翻回來了")
    assert facts["F-explore"][2] == 3 * (60 + 60 + 80 + 40 + 30 + 40 + 40)  # what it really walked
    assert [e[3] for e in episodes if e[0] == "walk"][:2] == [
        "自己走走 2 次，約 1.2 公尺",
        "自己走走 2 次，約 1.2 公尺",
    ]
    lines = [line for day in diary for line in day[3]]
    for text in (
        "跌倒後我試了好幾次才翻回來。",
        "跌倒後我自己一下就翻回來了。",
        "跌倒後我怎麼翻都翻不回來，等人來扶我。",
        "跌倒後還沒自己翻，就有人把我扶起來了。",
    ):
        assert text in lines


def test_a_migrated_log_with_both_shapes_distills_to_the_v4_memory():
    """A duck moved to the new engine with its memories (WP13) keeps v4 records from before."""
    for day in (1, 2):
        mixed = live_a_life(False, migrated_on_day=day)
        assert memory_view(mixed) == memory_view(live_a_life(False))
        mixed = live_a_life(False, migrated_on_day=day)
        assert memory_view(mixed, forget_falls=True) == memory_view(
            live_a_life(False), forget_falls=True
        )


def test_forgetting_a_v4_1_fall_takes_its_posture_record_too():
    """v4's fall record carried the cut walk; forgetting it made the walk count in full. A
    v4_1 fall's cut is on Duck Core's duck.posture(fallen), which goes with its percept."""
    old, new = live_a_life(False), live_a_life(True)
    facts, episodes, diary = memory_view(new, forget_falls=True)
    assert (facts, episodes, diary) == memory_view(old, forget_falls=True)
    assert "F-fell" not in facts
    assert facts["F-explore"][2] == 3 * (60 + 100 + 80 + 120 + 90 + 50 + 40)  # as if it never fell


def test_stale_results_and_righting_outcomes():
    from duckmind.duckcore import engine
    from duckmind.duckcore.memory import RIGHTING_TRIES, righting_outcome

    assert getattr(engine, "RIGHTING_TRIES", RIGHTING_TRIES) == RIGHTING_TRIES
    life = Life(True)
    life.stand()
    life.walk(100, 5, fall_after=2)
    life.righting([False, False, False])
    life.helped(0)
    life.stale_results()
    outcomes = [righting_outcome(event) for event in life.log.events]
    assert [o for o in outcomes if o] == ["stuck", "helped"]
    facts, episodes, _ = memory_view(life)
    assert facts["F-fell"][4] == 1  # the stale fall is not a fall
    assert len([e for e in episodes if e[0] == "walk"]) == 1  # a stale stand starts nothing


def test_every_new_record_kind_is_described_in_words():
    from duckmind.duckcore.narrate import category, describe

    life = live_a_life(True)
    life.act("duck.dock", bel_err_mm=120)
    life.act(
        "duck.look", kind="seek_nest", want="nest", bel_pose={"x_mm": 0, "y_mm": 0, "yaw_mrad": 0}
    )
    life.act("duck.stop", why="picked_up")
    life.emit("duck.act_timeout", {"act_id": "A1", "cmd": "duck.dock", "deadline_ms": 0})
    life.percept("body.dock", contact=True, via="pg", tries=1, seen_tag=True, act_id="A2")
    life.percept(
        "sense.sighting", what="nest", b_mrad=-140, r_mm=380, r_src="tag", conf_ppm=1_000_000,
        look_id="L4", edge="new", act_id="A3",
    )  # fmt: skip
    life.percept("sense.scan", act_id="A3", head_mrad=[], light="bright", sectors=[], sightings=[])
    life.percept("body.posture", want="stand", got="sit", ms=0, fell=False, act_id="A5")
    life.percept("body.play", policy="walk", meters=4, minute=1)
    life.percept("body.play", policy="walk", meters=3, minute=3, skill="bow")
    life.percept(
        "sense.touch",
        where="crown",
        seconds=1,
        firmness="light",
        path="ear",
        latency_ms=1100,
        legacy_phantom=True,
    )
    life.emit("duck.struggle", {"at": [0, 0, 0]})
    life.emit(
        "duck.heard",
        {"counts": {"voice": 2, "person": 1}, "room": "living", "since": None, "tv_on": False,
         "counted": ["person", "voice"], "sources": {"person": "camera", "voice": "ear"}},
    )  # fmt: skip
    kinds = set()
    for event in life.log.events:
        text = describe(event)
        assert text and text != event.kind, event.kind
        assert category(event), event.kind
        kinds.add(event.kind)
    assert {
        "duck.act", "duck.act_timeout", "walk.done", "body.fell", "body.righting", "body.posture",
        "body.dock", "sense.sighting", "sense.scan", "body.play", "duck.struggle", "duck.heard",
        "duck.righting_end", "duck.posture",
    } <= kinds  # fmt: skip
    texts = {describe(event) for event in life.log.events}
    assert "站起來了" in texts and "想站起來，但沒站穩，先坐著" in texts
    assert "看到窩的標籤了（38 公分遠）" in texts
    assert "走路第 3 秒跌倒了（在角落，1.9, 1.9 m），側躺著，等一下會試著翻回來" in texts
    old = live_a_life(False)  # the same sentence a v4 duck's fall record got
    assert [describe(e) for e in old.log.events if e.kind == "body.fell"] == [
        describe(e)
        for e in life.log.events
        if e.kind == "duck.posture" and e.payload["reason"] == "fell"
    ]
    assert any("靠耳朵、鏡頭" in text for text in texts)


def _remembered(session, forget: bool) -> tuple:
    """The duck's memory (facts, episodes, diary), and with ``forget`` what forgetting its
    falls, avoided spots and walks does (event ids aside: v4_1 logs have more records)."""
    core = session.core
    undone = []
    if forget:
        for fact_id in sorted(core.memory().facts):
            if fact_id in {"F-fell", "F-explore"} or fact_id.startswith("F-avoid-"):
                if fact_id in core.memory().facts:
                    result = dispatch(session, "forget", {"fact_id": fact_id})
                    undone.append(
                        (
                            fact_id,
                            result["rolled_back"],
                            result["removed_events"],
                            result["still_there"],
                        )
                    )
        spots = {
            zone: {key: value for key, value in spot.items() if key != "event"}
            for zone, spot in core.s.avoid_spots.items()
        }
        undone.append((dict(core.s.prefs), spots, core.s.bond_owner))
    memory = core.memory()
    facts = {
        fact_id: (fact.category, fact.text, fact.value, fact.strength, len(fact.evidence))
        for fact_id, fact in memory.facts.items()
    }
    episodes = [(e.kind, e.start, e.end, e.summary, e.salience) for e in memory.episodes]
    diary = [
        (d.date, d.positive, d.negative, [line.text for line in d.lines]) for d in memory.diary
    ]
    return facts, episodes, diary, undone


@pytest.mark.parametrize(
    "config,live",
    [
        ({"seed": 11, "nest": False}, {"days": 12, "routine": "daily"}),
        (
            {
                "seed": 111,
                "nest": True,
                "camera": True,
                "household": "dual_income",
                "battery_help": False,
            },
            {"family": "dual_income", "days": 12},
        ),
    ],
    ids=["floor", "nest"],
)
def test_a_v4_1_duck_remembers_what_its_v4_0_twin_remembers(config, live):
    """The same duck on the frozen v4 engine and on v4_1 + LegacyWorld (falls 800x as often):
    the same memory, and forgetting takes back the same things."""
    from duckmind.duckcore.api import open_session

    stress = {"fall_ppm_per_s": 40000, "righting_ppm": 150000, "fall_ppm_per_start": 20000}
    twins = []
    for rev in ("v4_0", "v4_1"):
        session = open_session({**config, **stress, "engine_rev": rev})
        dispatch(session, "live", dict(live))
        assert sum(event.kind == "body.fell" for event in session.core.log.events) >= 3
        twins.append((_remembered(session, forget=False), _remembered(session, forget=True)))
    assert twins[0] == twins[1]
