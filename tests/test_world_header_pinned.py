"""A v4_1 duck is Duck Core *and* its world (phase-4 review, finding 3; design A5, A6).

LegacyWorld draws with the duck's own key, so a saved v4_1 duck opens only with a world header
that is its own: its seed, the content table it hatched with, its camera as the nest marker, its
ear and its starting mode. Anything else would silently be another duck, so it is refused. The
header of an old v4 save is built on two sides (the api for Duck Core, LegacyWorld for itself):
they agree. And there is one list of engine revisions.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

import duckmind.world.legacy as legacy
from duckmind import engines
from duckmind.duckcore import api, content, engine


def _good() -> tuple[dict[str, Any], dict[str, Any]]:
    session = api.open_session({"engine_rev": "v4_1", "seed": 7, "camera": True})
    assert isinstance(session, api.Session) and session.world is not None
    return api.config_json(session), dict(session.world)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"seed": 8}, "seed 8"),
        ({"table_hash": "0123456789abcdef"}, "content table"),
        ({"marker": False}, "nest marker"),
        ({"sensors": {"ear": False, "conservative": True}}, "sensors.ear"),
        ({"sensors": {"ear": True, "conservative": False}}, "sensors.conservative"),
    ],
)
def test_a_world_header_that_is_not_the_ducks_is_refused(change: dict[str, Any], says: str) -> None:
    config, header = _good()
    assert api.open_session(config, header).world == header  # its own opens
    with pytest.raises(ValueError, match=says):
        api.open_session(config, {**header, **change})
    with pytest.raises(ValueError, match="not this duck's"):
        api.restore({"config": config, "world": {**header, **change}, "inputs": []})


def test_a_header_without_the_optional_mode_still_opens() -> None:
    config, header = _good()
    sensors = {k: v for k, v in header["sensors"].items() if k != "conservative"}
    assert api.open_session(config, {**header, "sensors": sensors}).world is not None


def test_the_api_and_legacyworld_build_the_same_header_from_an_old_config() -> None:
    """Design §5 puts the old-save conversion on the world's side; the api builds the header
    for Duck Core without importing the world (rule R1). The two must agree for every mix of
    the fields they read."""
    fields = api.v4_world_fields()
    for camera, ear, conservative, rates in itertools.product(
        (True, False),
        (True, False),
        (True, False),
        ({}, {"fall_ppm_per_s": 150_000}, {"dock_ppm": 700_000, "stand_ppm": 600_000}),
    ):
        config = api.Config(seed=31, camera=camera, ear=ear, conservative=conservative)
        v4_config = {
            "seed": 31,
            "camera": camera,
            "ear": ear,
            "conservative": conservative,
            **{name: rates.get(name, value) for name, value in fields.items()},
        }
        mine = api.legacy_world(config, rates)
        theirs = legacy.header_from_v4_config(v4_config, table_hash=content.table_hash())
        assert mine == theirs, (camera, ear, conservative, rates)


def test_there_is_one_list_of_engine_revisions() -> None:
    assert engine.ENGINE_REV == engines.CURRENT
    assert api.ENGINE_REVS == engines.REVS and api.FROZEN_REV == engines.OLD_SAVE
    assert set(engines.FROZEN) == {api.FROZEN_REV}
    assert legacy.HEADER_V == api.LEGACY_WORLD_V
