"""The page can always come back to the life duck (2026-10-10: the picker offered four demo ducks and
no way back to 蛋黃 once one was chosen): the picker lists the server's own life duck first, and
choosing it switches to that duck at its growth day now."""

from __future__ import annotations

from typing import Any

from duckmind.simweb import neuro_api
from duckmind.simweb.neuro_api import LIFE_PRESET, DuckChoice, NeuroHub, StubSource


def make_hub(gd: list[int]) -> NeuroHub:
    hub = NeuroHub(enabled=True, choice=neuro_api.PRESET_BY_ID["explorer"], given_name="")
    hub.attach(StubSource(hub.card()))
    hub.life_duck = lambda: DuckChoice(4107, gd[0], "蛋黃", LIFE_PRESET, "這隻鴨子自己的一生")
    return hub


def test_the_picker_lists_the_life_duck_first_with_its_name_and_growth_day() -> None:
    hub = make_hub([48])
    ducks: list[dict[str, Any]] = hub.ducks()
    assert [d["preset"] for d in ducks] == [
        LIFE_PRESET,
        "explorer",
        "snuggler",
        "dreamer",
        "hatchling",
    ]
    assert (ducks[0]["name"], ducks[0]["seed"], ducks[0]["gd"]) == ("蛋黃", 4107, 48)


def test_choosing_it_switches_to_the_life_duck_at_its_growth_day_now() -> None:
    gd = [48]
    hub = make_hub(gd)
    answer = hub.switch({"preset": "explorer"})
    assert answer["duck"]["gd"] == 60
    gd[0] = 49  # a midnight passed while the demo duck was on screen
    back = hub.switch({"preset": LIFE_PRESET})
    assert (back["duck"]["name"], back["duck"]["seed"], back["duck"]["gd"]) == ("蛋黃", 4107, 49)


def test_a_server_without_a_life_has_the_four_ducks_only() -> None:
    hub = NeuroHub(enabled=True, choice=neuro_api.PRESET_BY_ID["explorer"])
    assert [d["preset"] for d in hub.ducks()] == ["explorer", "snuggler", "dreamer", "hatchling"]
