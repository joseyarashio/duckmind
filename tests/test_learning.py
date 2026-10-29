from dataclasses import replace
from datetime import datetime, timedelta, timezone

from duckmind.core import DuckSelf, replay_events
from duckmind.learning import PreferenceLearner
from duckmind.provenance import Episode, Event, Outcome, OutcomeVerdict, SourceRef
from duckmind.store import SQLiteEventStore

NOW = datetime(2026, 9, 24, tzinfo=timezone.utc)


def event(name, kind, payload, source, at=NOW) -> Event:
    return Event(
        event_id=name,
        kind=kind,
        occurred_at=at,
        observed_at=at,
        source=SourceRef(kind=source, source_id="owner-or-sensor", reliability=1.0),
        trace_id="trace-learning",
        payload=payload,
    )


def episode_and_outcome(index=1, *, verdict=OutcomeVerdict.SUCCESS):
    invitation = event(
        f"invitation-{index}",
        "invitation",
        {"actor": "alex", "activity": "look_game"},
        "owner_request",
    )
    completion = event(
        f"completion-{index}",
        "skill_completed",
        {
            "activity": "look_game",
            "intent_id": f"intent-{index}",
            "status": "completed",
        },
        "skill_layer",
        NOW + timedelta(minutes=index, seconds=-1),
    )
    feedback = event(
        f"feedback-{index}",
        "explicit_feedback",
        {"actor": "alex", "activity": "look_game", "score": 1.0},
        "owner_feedback",
        NOW + timedelta(minutes=index),
    )
    episode = Episode(
        episode_id=f"episode-{index}",
        trace_id="trace-learning",
        started_at=invitation.occurred_at,
        ended_at=feedback.occurred_at,
        event_ids=(invitation.event_id, completion.event_id, feedback.event_id),
        intent_ids=(f"intent-{index}",),
        outcome_ids=(f"outcome-{index}",),
    )
    outcome = Outcome(
        outcome_id=f"outcome-{index}",
        episode_id=episode.episode_id,
        verdict=verdict,
        evaluator_version="owner-feedback-v1",
        evidence_event_ids=(
            invitation.event_id,
            completion.event_id,
            feedback.event_id,
        ),
        metrics={"feedback": 1.0},
    )
    return (invitation, completion, feedback), episode, outcome


def test_explicit_feedback_commits_one_bounded_preference_change(tmp_path) -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=NOW)
    events, episode, outcome = episode_and_outcome()
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        for entry in events:
            store.append_event(entry)
        store.commit_episode(episode, outcome)
        learner = PreferenceLearner(store)
        proposed = learner.propose(episode=episode, outcome=outcome, evidence=events, state=initial)
        assert proposed is not None
        assert 0 < proposed.after - proposed.before <= 0.005 + 1e-12

        committed = learner.commit(proposed, episode=episode, outcome=outcome, at=episode.ended_at)
        assert store.count("learning_updates") == 1
        assert store.count("events") == 4
        assert committed.status.value == "committed"
        assert (
            learner.propose(episode=episode, outcome=outcome, evidence=events, state=initial)
            is None
        )

        grown = replay_events(initial, [entry.event for entry in store.list_events()])
        assert grown.preferences["preferences.alex.look_game"] == committed.after
        assert replay_events(initial, [entry.event for entry in store.list_events()]) == grown


def test_unknown_or_untrusted_feedback_never_trains(tmp_path) -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=NOW)
    events, episode, outcome = episode_and_outcome(verdict=OutcomeVerdict.UNKNOWN)
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        assert (
            PreferenceLearner(store).propose(
                episode=episode, outcome=outcome, evidence=events, state=initial
            )
            is None
        )

    bad_feedback = Event(
        event_id=events[2].event_id,
        kind=events[2].kind,
        occurred_at=events[2].occurred_at,
        observed_at=events[2].observed_at,
        source=SourceRef(kind="llm_guess", source_id="llm", reliability=1.0),
        trace_id=events[2].trace_id,
        payload=events[2].payload,
    )
    with SQLiteEventStore(tmp_path / "untrusted.sqlite3") as store:
        assert (
            PreferenceLearner(store).propose(
                episode=episode,
                outcome=Outcome(
                    outcome_id=outcome.outcome_id,
                    episode_id=outcome.episode_id,
                    verdict=OutcomeVerdict.SUCCESS,
                    evaluator_version=outcome.evaluator_version,
                    evidence_event_ids=outcome.evidence_event_ids,
                    metrics=outcome.metrics,
                ),
                evidence=(events[0], events[1], bad_feedback),
                state=initial,
            )
            is None
        )


def test_daily_plasticity_cap_blocks_second_same_day_update(tmp_path) -> None:
    initial = DuckSelf.new(identity="duck-a", born_at=NOW)
    with SQLiteEventStore(tmp_path / "duckmind.sqlite3") as store:
        learner = PreferenceLearner(store)
        for index in (1, 2):
            events, episode, outcome = episode_and_outcome(index)
            for entry in events:
                store.append_event(entry)
            store.commit_episode(episode, outcome)
            proposed = learner.propose(
                episode=episode, outcome=outcome, evidence=events, state=initial
            )
            if index == 1:
                assert proposed is not None
                learner.commit(proposed, episode=episode, outcome=outcome, at=episode.ended_at)
                initial = replay_events(initial, [entry.event for entry in store.list_events()])
            else:
                assert proposed is None


def test_forged_feedback_object_with_real_event_id_cannot_train(tmp_path) -> None:
    events, episode, outcome = episode_and_outcome()
    forged = replace(
        events[2],
        payload={"actor": "alex", "activity": "look_game", "score": 1.0, "note": "forged"},
    )
    with SQLiteEventStore(tmp_path / "pet.sqlite3") as store:
        for entry in events:
            store.append_event(entry)
        store.commit_episode(episode, outcome)
        assert (
            PreferenceLearner(store).propose(
                episode=episode,
                outcome=outcome,
                evidence=(events[0], events[1], forged),
                state=DuckSelf.new(identity="duck", born_at=NOW),
            )
            is None
        )
