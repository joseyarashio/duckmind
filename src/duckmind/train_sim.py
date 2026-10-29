"""Deterministic cognitive-learning sandbox; no robot or ONNX motor retraining.

The official MuJoCo body gate lives in `sim_live`. This separate, much faster
virtual-days lab tests whether lived evidence causally changes future choices.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from duckmind.behavior import BehaviorChoice, choose_invitation
from duckmind.learning import PreferenceLearner
from duckmind.provenance import Episode, Event, Outcome, OutcomeVerdict, SourceRef
from duckmind.runtime import DuckRuntime
from duckmind.versions import package_fingerprint

START = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
PREFERENCE = "preferences.alex.look_game"


def _event(
    *,
    event_id: str,
    kind: str,
    at: datetime,
    source_kind: str,
    trace_id: str,
    payload: dict[str, object],
) -> Event:
    return Event(
        event_id=event_id,
        kind=kind,
        occurred_at=at,
        observed_at=at,
        source=SourceRef(kind=source_kind, source_id="virtual-owner", reliability=1.0),
        trace_id=trace_id,
        payload=payload,
    )


def _run_duck(path: Path, *, name: str, days: int, positive_feedback: bool) -> dict:
    trace_id = f"virtual-twin-{name}"
    engaged_days = 0
    decisions: list[str] = []
    with DuckRuntime(path, identity=name, born_at=START) as runtime:
        runtime.ingest(
            _event(
                event_id="owner-enable",
                kind="owner_play_enabled",
                at=START,
                source_kind="owner_control",
                trace_id=trace_id,
                payload={},
            )
        )
        learner = PreferenceLearner(runtime.store)
        for day in range(days):
            at = START + timedelta(days=day, minutes=1)
            invitation = _event(
                event_id=f"invitation-{day}",
                kind="invitation",
                at=at,
                source_kind="owner_request",
                trace_id=trace_id,
                payload={"actor": "alex", "activity": "look_game"},
            )
            runtime.ingest(invitation)
            decision = choose_invitation(runtime.state, invitation)
            decisions.append(decision.choice.value)
            engaged_days += decision.choice is BehaviorChoice.ENGAGE
            decision_event = _event(
                event_id=f"decision-{day}",
                kind="behavior_decision",
                at=at + timedelta(seconds=1),
                source_kind="system1",
                trace_id=trace_id,
                payload={
                    "goal": "play:alex:look_game"
                    if decision.choice is BehaviorChoice.ENGAGE
                    else None,
                    "invitation_event_id": invitation.event_id,
                    "choice": decision.choice.value,
                    "score": decision.score,
                    "influences": sorted(decision.influences),
                },
            )
            runtime.ingest(decision_event)
            evidence = [invitation, decision_event]
            verdict = OutcomeVerdict.UNKNOWN
            metrics = {}
            # The virtual world records execution separately from intent.
            # A decision alone never counts as a completed interaction.
            if decision.choice is BehaviorChoice.ENGAGE:
                completed = _event(
                    event_id=f"skill-completed-{day}",
                    kind="skill_completed",
                    at=at + timedelta(seconds=2),
                    source_kind="virtual_skill_oracle",
                    trace_id=trace_id,
                    payload={
                        "activity": "look_game",
                        "intent_id": f"intent-{day}",
                        "status": "completed",
                    },
                )
                runtime.ingest(completed)
                evidence.append(completed)
            # Feedback is counted only after a completed virtual game. Praise
            # for observing is not evidence of successful play.
            if positive_feedback and decision.choice is BehaviorChoice.ENGAGE:
                feedback = _event(
                    event_id=f"feedback-{day}",
                    kind="explicit_feedback",
                    at=at + timedelta(seconds=3),
                    source_kind="owner_feedback",
                    trace_id=trace_id,
                    payload={"actor": "alex", "activity": "look_game", "score": 1.0},
                )
                runtime.ingest(feedback)
                evidence.append(feedback)
                verdict = OutcomeVerdict.SUCCESS
                metrics = {"feedback": 1.0}
            episode = Episode(
                episode_id=f"episode-{day}",
                trace_id=trace_id,
                started_at=invitation.occurred_at,
                ended_at=evidence[-1].occurred_at,
                event_ids=tuple(item.event_id for item in evidence),
                intent_ids=(f"intent-{day}",),
                outcome_ids=(f"outcome-{day}",),
                runtime_versions={
                    "duckmind_rule": "preference-v1",
                    "scenario": "virtual-twins-v1",
                },
            )
            outcome = Outcome(
                outcome_id=f"outcome-{day}",
                episode_id=episode.episode_id,
                verdict=verdict,
                evaluator_version="virtual-feedback-v1",
                evidence_event_ids=tuple(item.event_id for item in evidence),
                metrics=metrics,
                reason=None if verdict is OutcomeVerdict.SUCCESS else "no_completed_game_feedback",
            )
            runtime.store.commit_episode(episode, outcome)
            proposal = learner.propose(
                episode=episode,
                outcome=outcome,
                evidence=evidence,
                state=runtime.state,
            )
            if proposal is not None:
                learner.commit(
                    proposal, episode=episode, outcome=outcome, at=evidence[-1].occurred_at
                )
                runtime.refresh()
        return {
            "final_preference": runtime.state.preferences.get(PREFERENCE, 0.5),
            "personality_sociability": runtime.state.personality.sociability,
            "engaged_days": engaged_days,
            "observed_days": decisions.count("observe"),
            "learning_updates": runtime.store.count("learning_updates"),
            "event_count": runtime.store.count("events"),
            "decision_trace": decisions,
        }


def run_twin_experiment(*, days: int = 30) -> dict[str, object]:
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")
    source_hash = package_fingerprint()
    with tempfile.TemporaryDirectory(prefix="duckmind-twins-") as directory:
        root = Path(directory)
        duck_a = _run_duck(root / "a.sqlite3", name="duck-a", days=days, positive_feedback=True)
        duck_b = _run_duck(root / "b.sqlite3", name="duck-b", days=days, positive_feedback=False)
        ablation = _run_duck(
            root / "ablation.sqlite3",
            name="duck-a-ablation",
            days=days,
            positive_feedback=False,
        )
    return {
        "scenario": "virtual-twins-v1",
        "duckmind_source_sha256": source_hash,
        "days": days,
        "seed": 0,
        "duck_a": duck_a,
        "duck_b": duck_b,
        "ablation": ablation,
        "causal_preference_delta": duck_a["final_preference"] - ablation["final_preference"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_twin_experiment(days=args.days)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
