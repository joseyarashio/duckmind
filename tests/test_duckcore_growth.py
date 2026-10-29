from duckmind.duckcore import content
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.duckcore.engine import DuckCore


def _credits(core: DuckCore) -> list[str]:
    return [event.payload["credit"] for event in core.log.events if event.kind == "growth.gd"]


def test_a_day_with_fifteen_awake_minutes_is_a_full_growth_day():
    core = DuckCore(seed=5)
    core.power_on("owner")
    core.advance(20)
    core.power_off("owner")
    core.advance(24 * 60)
    assert _credits(core)[0] == "full"
    assert core.s.gd2 >= 2


def test_days_off_give_half_credit_capped_at_five_growth_days_per_absence():
    core = DuckCore(seed=5)
    core.power_on("owner")
    core.advance(20)
    core.power_off("owner")
    core.advance(20 * 24 * 60)
    credits = _credits(core)
    assert credits[0] == "full"
    assert credits[1:11] == ["half"] * 10
    assert set(credits[11:]) == {"none"}
    assert core.s.gd == 6  # 1 full day + 10 half days


def test_the_absence_cap_resets_after_a_full_day():
    core = DuckCore(seed=5)
    for _ in range(2):
        core.power_on("owner")
        core.advance(20)
        core.power_off("owner")
        core.advance(12 * 24 * 60)
    credits = _credits(core)
    assert credits.count("full") == 2
    assert credits.count("half") == 20


def test_growth_pauses_without_verified_time():
    session = Session(Config(seed=3))
    dispatch(session, "set_setting", {"key": "ntp_ok", "value": False})
    dispatch(session, "live", {"days": 5, "routine": "daily"})
    assert session.core.s.gd2 == 0
    assert set(_credits(session.core)) == {"paused"}


def test_routines_grow_at_their_own_pace():
    grown = {}
    for routine in ("daily", "weekends", "neglect"):
        session = Session(Config(seed=11))
        dispatch(session, "live", {"days": 28, "routine": routine})
        assert session.core.log.verify()
        grown[routine] = session.core.s.gd
    assert grown["daily"] > grown["weekends"] > 0
    assert grown["daily"] > grown["neglect"] > 0


def test_stage_thresholds():
    assert content.stage_for(0).stage_id == content.STAGES[0].stage_id
    assert [stage.gd_min for stage in content.STAGES] == [0, 10, 45, 180]
    assert content.stage_for(44).gd_min == 10
    assert content.stage_for(45).gd_min == 45
    assert content.stage_for(400).gd_min == 180
