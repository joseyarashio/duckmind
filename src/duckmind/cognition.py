"""System 1/System 2 proposal boundary and grounding gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroundedFact:
    fact_id: str
    statement: str


@dataclass(frozen=True, slots=True)
class CognitiveProposal:
    intent: str
    confidence: float
    fact_ids: tuple[str, ...]
    planner: str


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    accepted: bool
    route: str
    reason: str | None = None


def route_proposal(
    proposal: CognitiveProposal,
    *,
    facts: tuple[GroundedFact, ...],
    system2_threshold: float,
) -> RoutingDecision:
    """Validate provenance, then route by confidence and declared planner.

    System 2 may interpret facts but cannot invent provenance identifiers. A
    low-confidence System 1 proposal is escalated; it is never auto-executed.
    """

    known = {fact.fact_id for fact in facts}
    for fact_id in proposal.fact_ids:
        if fact_id not in known:
            return RoutingDecision(False, "reject", f"ungrounded_fact:{fact_id}")
    if not 0.0 <= proposal.confidence <= 1.0:
        return RoutingDecision(False, "reject", "confidence_out_of_range")
    if proposal.planner == "system1" and proposal.confidence < system2_threshold:
        return RoutingDecision(False, "system2", "low_system1_confidence")
    if proposal.planner not in {"system1", "system2"}:
        return RoutingDecision(False, "reject", "unknown_planner")
    return RoutingDecision(True, proposal.planner)
