from duckmind.duckcore import commands
from duckmind.duckcore.api import Config, Session, dispatch
from duckmind.duckcore.engine import NAME_LEARN_DAYS
from duckmind.duckcore.narrate import describe


def _day(session: Session, *, calls: int, tv: bool = False) -> list[str]:
    """One evening visit with some calls of the duck's name; returns the name reactions."""
    core = session.core
    if not core.s.awake:
        dispatch(session, "act", {"action": "power_on", "persona": "owner"})
    dispatch(session, "act", {"action": "tv", "params": {"on": tv}})
    start = len(core.log.events)
    for _ in range(calls):
        dispatch(session, "act", {"action": "call_name", "persona": "child"})
        dispatch(session, "advance", {"minutes": 1})
    reactions = [
        event.payload["verb"]
        for event in core.log.events[start:]
        if event.kind == "duck.expression" and event.payload.get("reason") == "name"
    ]
    dispatch(session, "advance", {"minutes": 20})
    dispatch(session, "act", {"action": "power_off", "persona": "owner"})
    dispatch(session, "advance", {"minutes": 23 * 60})
    return reactions


def test_without_the_name_model_the_duck_cannot_hear_its_name():
    session = Session(Config(seed=5, name_ear=False))
    result = dispatch(session, "act", {"action": "call_name", "persona": "child"})
    assert result["perceived"] is False
    assert not [event for event in session.core.log.events if event.kind == "sense.name_heard"]


def test_hearing_its_name_on_several_days_teaches_it_its_name():
    session = Session(Config(seed=5, name="Pip"))
    first = _day(session, calls=2)
    assert first and set(first) == {"look_around"}  # "was that me?"
    for _ in range(NAME_LEARN_DAYS - 1):
        _day(session, calls=2)
    milestone = session.core.s.milestones["learned_name"]
    assert "Pip" in milestone["text"]
    later = _day(session, calls=2)
    assert later and set(later) == {"perk_tilt"}  # it knows it is being called
    assert "F-name" in session.core.memory().facts


def test_calls_with_the_tv_on_never_teach_the_name():
    session = Session(Config(seed=5))
    for _ in range(NAME_LEARN_DAYS + 1):
        _day(session, calls=2, tv=True)
    assert "learned_name" not in session.core.s.milestones
    assert "F-name" not in session.core.memory().facts
    heard = [event for event in session.core.log.events if event.kind == "sense.name_heard"]
    assert heard and "不太確定" in describe(heard[0])


def test_only_the_fact_of_hearing_is_kept():
    session = Session(Config(seed=5))
    _day(session, calls=3)
    heard = [event for event in session.core.log.events if event.kind == "sense.name_heard"]
    assert len(heard) == 3
    assert all(set(event.payload) == {"tv_on"} for event in heard)


def test_the_command_language_can_call_the_name():
    (command,) = commands.parse("小孩叫牠的名字")
    assert (command.kind, command.method, command.persona) == ("act", "call_name", "child")
