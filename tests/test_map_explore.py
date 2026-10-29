"""Map explore (docs/design/neural-duck/MAP-EXPLORE-2026-10-12.md): what the duck knows of its home from its own map, the page's
coarse picture of it, the growth input from new places, and that a duck with the map off reports none of it."""

from __future__ import annotations

import math

from duckmind.neuro import lastmap as LM


def room(seen_x: float = 3.0) -> LM.LastMap:
    """A 4 m x 3 m room (walls at x = -1 and 3, y = -1.5 and 1.5), seen clear from x = -1 up to ``seen_x``."""
    m = LM.LastMap()
    for k in range(-10, 31):
        x = k * 0.1 + 0.05
        for y in (-1.45, 1.45):
            if x <= seen_x + 0.1:
                m._add(x, y, 4.0, 0.0)
    for k in range(-15, 15):
        y = k * 0.1 + 0.05
        m._add(-0.95, y, 4.0, 0.0)
        if seen_x >= 3.0:
            m._add(2.95, y, 4.0, 0.0)
    for i in range(-9, int(seen_x * 10) - (1 if seen_x >= 3.0 else 0)):
        for j in range(-13, 13):
            m._add(i * 0.1 + 0.05, j * 0.1 + 0.05, -2.0, 0.0)
    return m


def test_a_seen_room_is_all_known_and_a_half_seen_one_has_open_edges():
    m = room(seen_x=3.0)
    home = m.home_view()
    assert home["edges"] == 0 and home["share"] == 1.0
    assert home["known_m2"] > 9.0 and home["places"] >= 6
    half = room(seen_x=1.0).home_view()
    assert half["share"] is not None and 0.0 < half["share"] < 0.9
    assert half["edges"] > 0 and half["places"] < home["places"]
    assert LM.LastMap().home_view() == {"known_m2": 0.0, "places": 0, "edges": 0, "share": None}


def test_a_one_cell_hole_between_looks_is_not_unseen_ground():
    m = room(seen_x=3.0)
    del m.cells[m.cell_of(0.55, 0.55)]  # a ray gap
    assert m.home_view()["edges"] == 0


def test_the_page_picture_is_coarse_and_run_length_coded():
    m = room(seen_x=1.0)
    pic = m.coarse()
    assert pic["cell_m"] == LM.CELL_M * LM.COARSE_CELLS
    n = 0
    seen = set()
    for part in __import__("re").findall(r"([abcd])(\d+)", pic["rle"]):
        n += int(part[1])
        seen.add(part[0])
    assert n == pic["w"] * pic["h"]
    assert {"b", "c"} <= seen
    assert len(pic["rle"]) < 400
    assert LM.LastMap().coarse()["rle"] == ""


def test_the_brain_reports_what_it_knows_of_its_home_with_the_map_on():
    from duckmind.neuro.brain import NeuroBrain
    from duckmind.neuro.contract import BodySense, CameraSense, FreeSector, SensorFrame
    from duckmind.neuro.genome import preset

    def fr(t: int, k: int) -> SensorFrame:
        floor = tuple(
            FreeSector(bearing=b, width=0.13, free_m=1.0, end="wall") for b in (-0.2, 0.0, 0.2)
        )
        return SensorFrame(
            t_ms=t,
            body=BodySense(posture="stand", odom=(0.003 * k, 0.0, 0.0)),
            camera=CameraSense(floor=floor, floor_t_ms=t, usable=True),
        )

    on = NeuroBrain(preset("explorer"), 2, lastmap="soft")
    off = NeuroBrain(preset("explorer"), 2, lastmap="off")
    t = 1000
    for k in range(150):
        _c, tel_on = on.step(t, fr(t, k))
        _c, tel_off = off.step(t, fr(t, k))
        t += 20
    assert "home" in tel_on.learned and tel_on.learned["home"]["known_m2"] > 0.0
    assert "home" not in tel_off.learned


class _Tel:
    def __init__(self, learned: dict) -> None:
        self.learned = learned


def test_new_places_its_own_map_saw_are_growth_inputs_and_held_moments_are_not():
    from duckmind.life import experience as X

    e = X.Experiences()
    out: list = []
    e._explore(1000, 1_000_000, _Tel({"home": {"places": 3}}), out)
    assert out == [] and e._home_places == 3  # (what it knew when the watch began is not a find)
    e._explore(1010, 1_010_000, _Tel({"home": {"places": 5}}), out)
    assert out == [{"k": "explore", "t": 1010}] and e.places_found == 2
    # within the throttle: absorbed, not told
    e._explore(1020, 1_020_000, _Tel({"home": {"places": 6}}), out)
    assert len(out) == 1 and e.places_found == 2
    # picked up a moment ago: not a real find
    e._lifted_ms = 1_400_000
    e._explore(1405, 1_405_000, _Tel({"home": {"places": 8}}), out)
    assert len(out) == 1 and e.ignored["explore_held"] == 1 and e._home_places == 8
    e._explore(1500, 1_500_000, _Tel({"home": {"places": 9}}), out)
    assert out[-1] == {"k": "explore", "t": 1500} and e.places_found == 3


def test_without_a_map_the_explore_input_is_what_it_was():
    from duckmind.life import experience as X

    e = X.Experiences()
    out: list = []
    e._explore(1000, 1_000_000, _Tel({"places": [1]}), out)
    e._explore(1400, 1_400_000, _Tel({"places": [1, 2]}), out)
    assert out == [{"k": "explore", "t": 1400}] and e.places_found == 0 and e._home_places is None


def test_the_growth_view_shows_what_it_knows_of_its_home():
    from duckmind.life import growth_view as GV

    assert GV._home(None, 0) is None
    v = GV._home({"known_m2": 12.345, "places": 9, "edges": 4, "share": 0.637}, 2)
    assert v == {"known_m2": 12.3, "places": 9, "share_pct": 64, "found": 2}
    assert (
        GV._home({"known_m2": 0.0, "places": 0, "edges": 0, "share": None}, 0)["share_pct"] is None
    )


def test_the_page_gets_the_ducks_own_map_laid_where_the_body_truly_is():
    import threading
    from types import SimpleNamespace

    from duckmind.neuro.contract import BodySense, CameraSense, SensorFrame
    from duckmind.simweb.neuro_live import NeuroDriver

    m = room(seen_x=1.0)
    m.corr = (0.0, 0.0, 0.0)
    m.update(
        0,
        SensorFrame(
            t_ms=0, body=BodySense(posture="sit", odom=(0.5, 0.0, 0.0)), camera=CameraSense()
        ),
    )
    brain = SimpleNamespace(lastmap=m, home_known=lambda: {"places": 1})
    # the body truly stands at (2, 1) facing +y: the map frame's origin is then at (2, 0.5), turned a quarter
    half = math.sqrt(0.5)
    body = SimpleNamespace(latest={"trunk": [2.0, 1.0, 0.3], "quat": [half, 0.0, 0.0, half]})
    fake = SimpleNamespace(
        lanes=SimpleNamespace(brain=brain), _tick_lock=threading.RLock(), _body=body
    )
    out = NeuroDriver.knows(fake)  # type: ignore[arg-type]
    assert out["on"] and out["home"] == {"places": 1} and "frontier" not in out
    assert out["grid"]["w"] > 0 and out["grid"]["rle"]
    w = out["world"]
    assert (
        abs(w["yaw"] - math.pi / 2) < 1e-3 and abs(w["x"] - 2.0) < 1e-3 and abs(w["y"] - 0.5) < 1e-3
    )
    off = SimpleNamespace(
        lanes=SimpleNamespace(
            brain=SimpleNamespace(lastmap=LM.LastMap(LM.LastMapConfig(enabled=False)))
        )
    )
    assert NeuroDriver.knows(off) == {"ok": True, "on": False}  # type: ignore[arg-type]


def test_the_minimap_offers_the_layer_in_six_languages():
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
    page = (static / "index.html").read_text(encoding="utf-8")
    assert 'id="miniMapKnows"' in page and "page.minimap.knows" in page
    for lang in ("zh-Hant", "en", "ja", "es", "zh-Hans", "ko"):
        text = (static / f"i18n.{lang}.page.js").read_text(encoding="utf-8")
        for key in (
            "page.minimap.knows",
            "page.growth.learn.home_line",
            "page.growth.exp.ignored.explore_held",
        ):
            assert f'"{key}"' in text, (lang, key)
