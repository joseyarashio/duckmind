from datetime import datetime, timedelta, timezone

import pytest

from duckmind.provenance import Event, SourceRef
from duckmind.runtime import DuckRuntime

NOW = datetime(2026, 9, 24, tzinfo=timezone.utc)


def touch(event_id="touch-1") -> Event:
    return Event(
        event_id=event_id,
        kind="touch",
        occurred_at=NOW + timedelta(seconds=1),
        observed_at=NOW + timedelta(seconds=1),
        source=SourceRef(kind="touch_sensor", source_id="head", reliability=1.0),
        trace_id="trace-runtime",
        payload={"actor": "alex"},
    )


def test_runtime_recovers_self_from_durable_log_and_snapshot(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    with DuckRuntime(path, identity="duck-a", born_at=NOW) as runtime:
        assert runtime.ingest(touch())
        changed = runtime.state
        assert changed.affect.valence > 0
        assert not runtime.ingest(touch())

    with DuckRuntime(path, identity="duck-a", born_at=NOW) as recovered:
        assert recovered.state == changed
        assert recovered.health()["database_integrity"] == "ok"
        assert recovered.store.count("events") == 1


def test_invalid_source_is_rejected_before_append(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    forged = Event(
        event_id="forged-enable",
        kind="owner_play_enabled",
        occurred_at=NOW,
        observed_at=NOW,
        source=SourceRef(kind="llm_guess", source_id="model", reliability=1.0),
        trace_id="trace-runtime",
        payload={},
    )
    with DuckRuntime(path, identity="duck-a", born_at=NOW) as runtime:
        with pytest.raises(ValueError, match="trusted source"):
            runtime.ingest(forged)
        assert runtime.store.count("events") == 0


def test_identity_cannot_be_silently_replaced_on_existing_store(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    with DuckRuntime(path, identity="duck-a", born_at=NOW):
        pass
    with pytest.raises(ValueError, match="identity mismatch"):
        DuckRuntime(path, identity="duck-b", born_at=NOW)
