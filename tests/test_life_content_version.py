"""Replay safety (docs/design/GROWTH-G1-2026-10-10.md, part 4): a v4_1 save carries the content
version it was written on, and a tuning change to ``content.py`` makes an older save say so
instead of replaying into another duck."""

from __future__ import annotations

import json

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld: a v4_1 duck lives in it)
from duckmind.duckcore import api, content
from duckmind.duckcore.api import (
    Config,
    SaveContentMismatch,
    Session,
    dispatch,
    restore,
    save_json,
)

BIRTH = "2026-10-11T09:30:00+08:00"


def _lived() -> Session:
    session = Session(Config(seed=4107, name="蛋黃", start=BIRTH))
    dispatch(session, "advance", {"seconds": 40 * 3600})
    return session


def test_a_save_carries_the_content_version_and_replays_to_the_same_duck() -> None:
    session = _lived()
    saved = json.loads(json.dumps(save_json(session), ensure_ascii=False))
    assert saved["content"] == {
        "engine_rev": api.ENGINE_REV,
        "version": content.CONTENT_VERSION,
        "table_hash": content.table_hash(),
    }
    again = restore(saved)
    assert again.core.log.head == session.core.log.head
    assert again.core.s.gd == session.core.s.gd == 2


def test_a_save_from_another_content_table_is_refused_in_plain_words_not_replayed() -> None:
    saved = json.loads(json.dumps(save_json(_lived()), ensure_ascii=False))
    saved["content"]["table_hash"] = "0123456789abcdef"
    with pytest.raises(SaveContentMismatch) as error:
        restore(saved)
    assert error.value.written["table_hash"] == "0123456789abcdef"
    assert error.value.now == api.content_version()
    saved["content"]["version"] = "duckcore-content-2"
    saved["content"]["table_hash"] = content.table_hash()
    with pytest.raises(SaveContentMismatch) as other:
        restore(saved)
    assert other.value.written["version"] == "duckcore-content-2"


def test_the_server_refuses_to_start_on_such_a_save_and_leaves_it_alone(tmp_path) -> None:
    from duckmind.simweb.server import SimState

    saved = json.loads(json.dumps(save_json(_lived()), ensure_ascii=False))
    saved["content"]["table_hash"] = "0123456789abcdef"
    path = tmp_path / "core.json"
    path.write_text(json.dumps(saved, ensure_ascii=False))
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exit_:
        SimState(Config(seed=4107), None, path)
    assert "另一個內容版本" in str(exit_.value) and "0123456789abcdef" in str(exit_.value)
    assert path.read_bytes() == before  # never overwritten


def test_a_real_tuning_change_to_content_is_caught_by_an_older_save(monkeypatch) -> None:
    saved = json.loads(json.dumps(save_json(_lived()), ensure_ascii=False))
    tuned = dict(content.SOCIAL)
    tuned["floor"] = int(tuned["floor"]) + 1  # one number changed in content.py
    monkeypatch.setattr(content, "SOCIAL", tuned)
    assert content.table_hash() != saved["content"]["table_hash"]
    with pytest.raises(SaveContentMismatch):
        restore(saved)
    # and the world header (an older save without the block) still guards it the old way
    saved.pop("content")
    with pytest.raises(ValueError, match="content table"):
        restore(saved)


def test_a_save_before_the_block_existed_still_opens() -> None:
    saved = json.loads(json.dumps(save_json(_lived()), ensure_ascii=False))
    saved.pop("content")
    assert restore(saved).core.s.gd == 2


def test_a_duck_without_a_chosen_birth_saves_exactly_as_before() -> None:
    session = Session(Config(seed=9))
    assert "start" not in save_json(session)["config"]
    assert session.core.s.now.isoformat() == "2026-12-20T19:00:00+08:00"
    born = Session(Config(seed=9, start=BIRTH))
    assert born.core.s.now.isoformat() == BIRTH and save_json(born)["config"]["start"] == BIRTH
