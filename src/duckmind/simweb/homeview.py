"""The home the simulator page draws (phase 2, physical-design spec §12–§13).

The page draws the room from the same layout the physics runs (:mod:`duckmind.world.homelayout`,
``Layout.page_json``: every wall, piece of furniture, toy and the nest in world millimetres),
plus the shapes the world adds to it — the visible hand, the family's bodies and the nest's
tags and backlight — so what is drawn is what the physics has. Which home: the one the live
body runs (``DUCKMIND_LIVE_HOME``, default ``default_home``; a duck without a nest gets the
``no_nest`` variant, owner decision A2). Without live physics, or on phase 0's body host, the
page still draws that home, but nothing of it is physical (``physical`` is decided by the
live world, see :meth:`duckmind.simweb.live.LiveBridge.world_view`).

:func:`home_payload` is what ``/api/home`` answers and what ``/api/body?home=<fingerprint>``
carries when the page's copy is not this one::

    {"name": "default_home,no_nest", "fingerprint": "...", "layout": {...page_json...},
     "hand": {"body": "kit_hand", "parts": [...]}, "people": {persona: {...}},
     "nest_parts": {...}, "aids": {...}, "error": null}
"""

from __future__ import annotations

import functools
from typing import Any

from duckmind.livebody import DEFAULT_HOME, live_home

HOME_SCHEMA = "simweb-home-1"

# The home this server was asked to run (``--home``) or the layout the editor applied (a layout file's
# path): None means DUCKMIND_LIVE_HOME, else default_home. One server, one home at a time.
_ASKED: str | None = None


def set_home(home: str | None) -> None:
    """The home the page draws and the physics runs (a name, ``name,variant`` or a layout file)."""
    global _ASKED
    _ASKED = home or None


def asked_home() -> str | None:
    return _ASKED


def page_home(*, nest: bool | None) -> str:
    """The home the page draws for a duck with (or without) a nest: the live body's home, or
    ``default_home`` (its ``no_nest`` variant for a duck without a nest) when the physics
    runs phase 0's body host or the flat floor (then it is only drawn, not physical). A layout file
    (the editor's, or ``--home FILE.json``) is a home as it is."""
    home = live_home(_ASKED, nest=nest)
    if home is None or home == "flat" or (home.endswith(".json") and not _ASKED):
        home = live_home(DEFAULT_HOME, nest=nest) or DEFAULT_HOME
    return home


def layout_of(home: str) -> Any:
    """The resolved layout of a home (cached by name or path)."""
    return _layout(home)


@functools.lru_cache(maxsize=8)
def _layout(home: str) -> Any:
    from duckmind.world import homelayout

    return homelayout.load(home)


@functools.lru_cache(maxsize=8)
def _payload(home: str) -> dict[str, Any]:
    from duckmind.world import hand, nestpad, people

    layout = _layout(home)
    return {
        "schema": HOME_SCHEMA,
        "name": home,
        "fingerprint": layout.fingerprint,
        "layout": layout.page_json(),
        "hand": hand.page_json(),
        "people": people.page_json(),
        "nest_parts": nestpad.page_json(layout.nest),
        # Drawn on the page only, never in the physics: markings to judge distances by eye.
        "aids": {"floor_dots_mm": [[500 * i, 0] for i in range(1, 5)], "leash_mm": 1500},
        "error": None,
    }


def home_payload(home: str) -> dict[str, Any]:
    """The page's home (see the module docstring); on a layout that cannot be read here, a
    payload without a layout that says why in plain words (the page keeps its floor)."""
    try:
        return _payload(home)
    except Exception as error:
        return {
            "schema": HOME_SCHEMA,
            "name": home,
            "fingerprint": None,
            "layout": None,
            "error": f"讀不到家的佈置（{home}）：{type(error).__name__}: {error}",
        }


@functools.lru_cache(maxsize=8)
def _home_lights(home: str) -> Any:
    from duckmind.world import lights

    return lights.HomeLights.from_layout(_layout(home))


def plan_in_world(home: str, x_m: float, y_m: float) -> tuple[float, float] | None:
    """Duck Core's plan point (its own frame, m) in the home's world (the layout's nest or mat
    is its origin), or None when the home cannot be read here."""
    try:
        frame = _layout(home).frame
    except Exception:
        return None
    wx, wy = frame.to_world(x_m * 1000.0, y_m * 1000.0)
    return wx / 1000.0, wy / 1000.0


def room_light(home: str, moment: Any, at_m: tuple[float, float] | None = None) -> Any:
    """The light of the room the duck is in (``at_m``: world metres; None: the home's first
    room) at the duck's local time, from the same layout and the same model the world's camera
    uses (world.lights, spec §11.3), or None when the home cannot be read here."""
    from duckmind.world import lights

    try:
        state = lights.home_light_state(moment, _home_lights(home))
    except Exception:
        return None
    if at_m is not None:
        return state.at(at_m[0], at_m[1])
    return next(iter(state.rooms.values()), None)


def home_brief(home: str) -> dict[str, Any]:
    """Which home, cheaply (every /api/call answer carries it)."""
    payload = home_payload(home)
    layout = payload.get("layout") or {}
    return {
        "name": home,
        "id": layout.get("id"),
        "zh": layout.get("zh"),
        "variant": layout.get("variant"),
        "fingerprint": payload.get("fingerprint"),
    }
