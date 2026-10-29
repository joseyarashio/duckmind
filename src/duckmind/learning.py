"""Evidence-gated, slowly plastic preference learning.

This adjusts behavior values, never motor policy weights or safety settings.
Unknown outcomes, implicit silence and LLM guesses are not rewards.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Sequence

from duckmind.core import DuckSelf
from duckmind.provenance import (
    Episode,
    Event,
    LearningStatus,
    LearningUpdate,
    Outcome,
    OutcomeVerdict,
    SourceRef,
)
from duckmind.store import SQLiteEventStore

RULE_VERSION = "preference-v1"
DAILY_MAX_ABS_DELTA = 0.005
EPISODE_MAX_ABS_DELTA = 0.005
LEARNING_RATE = 0.02
NAME = re.compile(r"[a-z0-9_-]+")
UPDATE_NAMESPACE = uuid.UUID("56ab02d7-c1de-4ccd-b9d9-0a1b443a8b28")


class PreferenceLearner:
    def __init__(self, store: SQLiteEventStore) -> None:
        self._store = store

    def propose(
        self,
        *,
        episode: Episode,
        outcome: Outcome,
        evidence: Sequence[Event],
        state: DuckSelf,
    ) -> LearningUpdate | None:
        if episode.ended_at is None or outcome.episode_id != episode.episode_id:
            return None
        if outcome.outcome_id not in episode.outcome_ids or outcome.verdict not in {
            OutcomeVerdict.SUCCESS,
            OutcomeVerdict.FAILURE,
        }:
            return None
        indexed = {event.event_id: event for event in evidence}
        if len(indexed) != len(evidence) or any(
            event_id not in indexed or event_id not in episode.event_ids
            for event_id in outcome.evidence_event_ids
        ):
            return None
        # Callers may hand us objects with familiar IDs but altered payloads.
        # Learning must use the durable event log, not caller-supplied copies.
        if any(self._store.event_by_id(item.event_id) != item for item in evidence):
            return None
        feedback = [
            indexed[event_id]
            for event_id in outcome.evidence_event_ids
            if indexed[event_id].kind == "explicit_feedback"
        ]
        if len(feedback) != 1:
            return None
        event = feedback[0]
        if event.source.kind != "owner_feedback" or event.source.reliability < 0.8:
            return None
        actor = event.payload.get("actor")
        activity = event.payload.get("activity")
        score = event.payload.get("score")
        if not isinstance(actor, str) or not NAME.fullmatch(actor):
            return None
        if not isinstance(activity, str) or not NAME.fullmatch(activity):
            return None
        invitations = [
            indexed[event_id]
            for event_id in outcome.evidence_event_ids
            if indexed[event_id].kind == "invitation"
        ]
        completions = [
            indexed[event_id]
            for event_id in outcome.evidence_event_ids
            if indexed[event_id].kind == "skill_completed"
        ]
        if len(invitations) != 1 or len(completions) != 1:
            return None
        invitation, completion = invitations[0], completions[0]
        if (
            invitation.source.kind != "owner_request"
            or invitation.source.reliability < 0.8
            or invitation.payload.get("actor") != actor
            or invitation.payload.get("activity") != activity
            or completion.source.kind not in {"skill_layer", "virtual_skill_oracle"}
            or completion.source.reliability < 0.8
            or (
                completion.source.kind == "virtual_skill_oracle"
                and episode.runtime_versions.get("scenario") != "virtual-twins-v1"
            )
            or completion.payload.get("status") != "completed"
            or completion.payload.get("activity") != activity
            or completion.payload.get("intent_id") not in episode.intent_ids
            or not invitation.occurred_at <= completion.occurred_at <= event.occurred_at
        ):
            return None
        if isinstance(score, bool) or score not in (0, 1):
            return None
        if outcome.metrics.get("feedback") != float(score):
            return None
        if (score == 1) != (outcome.verdict is OutcomeVerdict.SUCCESS):
            return None
        parameter = f"preferences.{actor}.{activity}"
        if self._store.has_learning_for_episode(episode.episode_id, parameter):
            return None
        current = state.preferences.get(parameter, 0.5)
        if not math.isfinite(current) or not 0 <= current <= 1:
            raise ValueError("current preference is invalid")
        already_used = self._store.daily_learning_magnitude(parameter, episode.ended_at.date())
        remaining = max(0.0, DAILY_MAX_ABS_DELTA - already_used)
        if remaining <= 1e-12:
            return None
        raw_delta = LEARNING_RATE * event.source.reliability * (float(score) - current)
        bound = min(EPISODE_MAX_ABS_DELTA, remaining)
        delta = min(bound, max(-bound, raw_delta))
        if abs(delta) <= 1e-12:
            return None
        update_id = str(
            uuid.uuid5(UPDATE_NAMESPACE, f"{episode.episode_id}:{parameter}:{RULE_VERSION}")
        )
        return LearningUpdate(
            update_id=update_id,
            episode_id=episode.episode_id,
            parameter=parameter,
            before=current,
            after=min(1.0, max(0.0, current + delta)),
            max_abs_delta=EPISODE_MAX_ABS_DELTA,
            rule_version=RULE_VERSION,
            outcome_ids=(outcome.outcome_id,),
            status=LearningStatus.PROPOSED,
        )

    def commit(
        self,
        proposal: LearningUpdate,
        *,
        episode: Episode,
        outcome: Outcome,
        at: datetime,
    ) -> LearningUpdate:
        if proposal.status is not LearningStatus.PROPOSED or proposal.rule_version != RULE_VERSION:
            raise ValueError("only a proposal from the current rule may be committed")
        if proposal.episode_id != episode.episode_id or proposal.outcome_ids != (
            outcome.outcome_id,
        ):
            raise ValueError("learning proposal has wrong episode/outcome")
        if episode.ended_at is None or at.tzinfo is None:
            raise ValueError("closed episode and timezone-aware commit time required")
        if at < episode.ended_at:
            raise ValueError("learning cannot be observed before the episode ends")
        committed = replace(proposal, status=LearningStatus.COMMITTED)
        event = Event(
            event_id=f"learning:{proposal.update_id}",
            kind="learning_committed",
            occurred_at=episode.ended_at,
            observed_at=at.astimezone(timezone.utc),
            source=SourceRef(kind="learning_engine", source_id=RULE_VERSION, reliability=1.0),
            trace_id=episode.trace_id,
            payload={
                "update_id": committed.update_id,
                "parameter": committed.parameter,
                "before": committed.before,
                "after": committed.after,
                "evidence_event_ids": list(outcome.evidence_event_ids),
            },
            parent_event_ids=outcome.evidence_event_ids,
        )
        self._store.commit_learning_with_event(committed, event)
        return committed
