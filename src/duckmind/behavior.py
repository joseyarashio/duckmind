"""Deterministic System 1 invitation selection with hard safety gates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from duckmind.core import DuckSelf
from duckmind.provenance import Event


class BehaviorChoice(StrEnum):
    DECLINE = "decline"
    OBSERVE = "observe"
    ENGAGE = "engage"
    CONTINUE = "continue"


@dataclass(frozen=True, slots=True)
class BehaviorDecision:
    choice: BehaviorChoice
    score: float
    reason: str
    invitation_event_id: str
    influences: frozenset[str]


def choose_invitation(state: DuckSelf, invitation: Event, *, seed: int = 0) -> BehaviorDecision:
    """Choose intent only. A separate skill layer may issue a body command."""

    if (
        invitation.kind != "invitation"
        or invitation.source.kind != "owner_request"
        or invitation.source.reliability < 0.8
    ):
        return BehaviorDecision(
            BehaviorChoice.DECLINE,
            0.0,
            "untrusted_invitation",
            invitation.event_id,
            frozenset(),
        )
    actor = invitation.payload.get("actor")
    activity = invitation.payload.get("activity")
    if not isinstance(actor, str) or not actor or not isinstance(activity, str) or not activity:
        return BehaviorDecision(
            BehaviorChoice.DECLINE,
            0.0,
            "invalid_invitation",
            invitation.event_id,
            frozenset(),
        )
    if not state.play_enabled:
        return BehaviorDecision(
            BehaviorChoice.DECLINE,
            0.0,
            "play_disabled",
            invitation.event_id,
            frozenset({"play_enabled"}),
        )
    if state.fallen or state.limp or state.battery_fraction < 0.25:
        return BehaviorDecision(
            BehaviorChoice.DECLINE,
            0.0,
            "body_unsafe",
            invitation.event_id,
            frozenset({"fallen", "limp", "battery_fraction"}),
        )
    if state.drives.rest_need >= 0.8:
        return BehaviorDecision(
            BehaviorChoice.DECLINE,
            0.0,
            "rest_needed",
            invitation.event_id,
            frozenset({"drives.rest_need"}),
        )
    goal = f"play:{actor}:{activity}"
    if state.last_decision_at is not None and state.current_goal is not None:
        age = (invitation.occurred_at - state.last_decision_at).total_seconds()
        if 0 <= age < 5:
            if state.current_goal == goal:
                return BehaviorDecision(
                    BehaviorChoice.CONTINUE,
                    1.0,
                    "same_goal_cooldown",
                    invitation.event_id,
                    frozenset({"current_goal", "last_decision_at"}),
                )
            return BehaviorDecision(
                BehaviorChoice.OBSERVE,
                0.0,
                "switch_cooldown",
                invitation.event_id,
                frozenset({"current_goal", "last_decision_at"}),
            )
    path = f"preferences.{actor}.{activity}"
    preference = state.preferences.get(path, 0.5)
    score = (
        0.20
        + 0.40 * preference
        + 0.15 * state.personality.sociability
        + 0.15 * state.drives.social_need
        - 0.20 * state.drives.rest_need
    )
    influences = frozenset(
        {path, "personality.sociability", "drives.social_need", "drives.rest_need"}
    )
    if score >= 0.49:
        return BehaviorDecision(
            BehaviorChoice.ENGAGE,
            score,
            "preference_and_needs",
            invitation.event_id,
            influences,
        )
    # A small, seeded exploration budget lets the duck gather real outcome
    # evidence without making every invitation a random personality flip.
    # The decision is stable for the same day/activity/seed and never overrides
    # the hard safety gates above.
    if score >= 0.44:
        choice_key = f"{seed}:{invitation.occurred_at.date()}:{actor}:{activity}".encode()
        if int.from_bytes(hashlib.sha256(choice_key).digest()[:4], "big") % 100 < 30:
            return BehaviorDecision(
                BehaviorChoice.ENGAGE,
                score,
                "seeded_exploration",
                invitation.event_id,
                influences | frozenset({"exploration_seed"}),
            )
    return BehaviorDecision(
        BehaviorChoice.OBSERVE,
        score,
        "low_social_value",
        invitation.event_id,
        influences,
    )
