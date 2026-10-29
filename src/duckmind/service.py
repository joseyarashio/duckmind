"""Local-only, fail-closed pet service for a supervised first-board slice.

The service owns the SQLite writer and exposes no network listener. Its socket
is mode 0600; commissioning uses a local sudo-authorised CLI. It never calls
robot.enable, loads a policy, or sends joint targets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from duckmind.behavior import BehaviorChoice, choose_invitation
from duckmind.body import BodyPort
from duckmind.learning import PreferenceLearner
from duckmind.pet_body_trace import BodyTraceRecorder
from duckmind.policy_approval import loaded_official_v5_walk, official_v5_walk_file_matches
from duckmind.provenance import Episode, Event, Outcome, OutcomeVerdict, SourceRef
from duckmind.robotd import RobotdBodyPort, RobotdSafetyClient
from duckmind.runtime import DuckRuntime
from duckmind.skills import (
    EmbodiedSkillLayer,
    SkillPhase,
    SkillStatus,
    front_gaze_observation_issue,
)
from duckmind.versions import package_fingerprint

FEEDBACK_WINDOW = timedelta(seconds=120)
NAME = re.compile(r"[a-z0-9_-]{1,32}")
RELEASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
MAX_REQUEST_BYTES = 8192
BOARD_RELEASE_ID_FILE = Path("/opt/duckmind/current/RELEASE_ID")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PetController:
    """Single event-loop owner of durable cognition and interaction evidence."""

    def __init__(
        self,
        runtime: DuckRuntime,
        *,
        robotd_socket: str | Path,
        backend: str,
        body_factory: Callable[[], RobotdBodyPort] | None = None,
        safety_factory: Callable[[], RobotdSafetyClient] | None = None,
        approve_real_v5_head_profile: bool = False,
        approved_release_id: str | None = None,
        release_id_path: Path = BOARD_RELEASE_ID_FILE,
        sim_body_trace: Path | None = None,
        policy_file_check: Callable[[], bool] | None = None,
    ) -> None:
        if backend not in {"real", "simulator"}:
            raise ValueError("backend must be real or simulator")
        if sim_body_trace is not None and backend != "simulator":
            raise ValueError("body trace is simulator only")
        self.runtime = runtime
        self.backend = backend
        self.robotd_socket = Path(robotd_socket)
        self.approve_real_v5_head_profile = approve_real_v5_head_profile
        self.approved_release_id = approved_release_id
        self.release_id_path = release_id_path
        self.sim_body_trace = sim_body_trace
        self._policy_file_check = policy_file_check or official_v5_walk_file_matches
        self._body_factory = body_factory or (
            lambda: RobotdBodyPort(self.robotd_socket, backend=backend, timeout_s=1.0)
        )
        self._safety_factory = safety_factory or (
            lambda: RobotdSafetyClient(self.robotd_socket, timeout_s=1.0)
        )
        self._command_lock = threading.Lock()
        self._cancel = threading.Event()
        self._active = False
        self._stops_in_flight = 0
        self._storage_fault: str | None = None
        self._stop_epoch = 0
        self._learner = PreferenceLearner(runtime.store)
        self._reconcile_seq = 0
        self._reconcile_pending: defaultdict[str, list[Event]] = defaultdict(list)
        # A process restart never silently resumes autonomous play.
        self._event(
            "body_lost",
            "runtime_safety",
            f"boot-passive:{uuid.uuid4()}",
            {"reason": "process_start"},
        )

    def _event(
        self,
        kind: str,
        source_kind: str,
        event_id: str,
        payload: dict[str, object],
        *,
        trace_id: str | None = None,
        parent_event_ids: tuple[str, ...] = (),
    ) -> Event:
        at = _now()
        event = Event(
            event_id=event_id,
            kind=kind,
            occurred_at=at,
            observed_at=at,
            source=SourceRef(source_kind, "duckmind-local-service", 1.0),
            trace_id=trace_id or event_id,
            payload=payload,
            parent_event_ids=parent_event_ids,
        )
        self.runtime.ingest(event)
        return event

    def _head_approval_state(self) -> str:
        """A stale systemd drop-in must not approve a newly selected release."""
        if self.backend != "real":
            return "not_required"
        if not self.approve_real_v5_head_profile:
            return "not_approved"
        release_id = self.approved_release_id
        if not isinstance(release_id, str) or RELEASE_ID.fullmatch(release_id) is None:
            return "release_not_bound"
        try:
            record = self.release_id_path
            if record.is_symlink() or not record.is_file() or record.stat().st_size > 128:
                return "release_record_invalid"
            if record.read_bytes() != f"{release_id}\n".encode("ascii"):
                return "release_mismatch"
        except (OSError, UnicodeError):
            return "release_record_invalid"
        return "approved"

    def _body_preflight(self) -> tuple[float, str, bool, bool]:
        approval = self._head_approval_state()
        if self.backend == "real" and approval != "approved":
            raise RuntimeError(f"real_v5_head_profile_{approval}")
        with self._safety_factory() as safety:
            health = safety.health()
            policies = safety.policies() if self.backend == "real" else None
        if health.get("healthy") is not True:
            raise RuntimeError(f"robotd_unhealthy:{health.get('reason') or 'unknown'}")
        if self.backend == "real" and not loaded_official_v5_walk(policies):
            raise RuntimeError("loaded_v5_walk_mismatch")
        battery = health.get("battery")
        if battery is None:
            if self.backend == "real":
                raise RuntimeError("battery_unknown")
            battery_fraction = 1.0  # simulator has no physical pack
        else:
            if not isinstance(battery, dict):
                raise RuntimeError("battery_malformed")
            percent = battery.get("percent")
            if (
                isinstance(percent, bool)
                or not isinstance(percent, (int, float))
                or not 0 <= percent <= 100
            ):
                raise RuntimeError("battery_malformed")
            battery_fraction = float(percent) / 100.0
        if self.backend == "real" and self._policy_file_check() is not True:
            raise RuntimeError("official_v5_walk_bytes_mismatch")
        with self._body_factory() as body:
            if isinstance(body, RobotdBodyPort) and not body.supports_v5_head_profile:
                raise RuntimeError("unsupported_robotd_head_profile")
            observation = body.observe()
        issue = front_gaze_observation_issue(observation)
        if issue is not None:
            raise RuntimeError(issue)
        if battery_fraction < 0.30:
            raise RuntimeError("battery_below_play_floor")
        return (
            battery_fraction,
            observation.policy,
            observation.safety.fallen,
            observation.safety.limp,
        )

    def _safe_stop(self) -> bool:
        # The command lock guarantees this is ordered after a concurrent gaze
        # dispatch. robot.stop halts locomotion intent, not servo torque/head.
        with self._command_lock:
            with self._safety_factory() as safety:
                return safety.stop().accepted

    def _run_gaze(self, interaction_id: str) -> SkillStatus:
        with self._body_factory() as body:
            recorder = (
                BodyTraceRecorder(
                    body,
                    interaction_id=interaction_id,
                    source_sha256=package_fingerprint(),
                )
                if self.sim_body_trace is not None
                else None
            )
            port: BodyPort = recorder if recorder is not None else body
            status = EmbodiedSkillLayer(port).perform_front_gaze(
                cancelled=self._cancel, command_lock=self._command_lock
            )
            if recorder is not None:
                assert self.sim_body_trace is not None
                recorder.save(self.sim_body_trace, status=status)
            return status

    async def _latch_body_lost(self, reason: str) -> None:
        self._cancel.set()
        self._stop_epoch += 1
        if self.runtime.state.play_enabled or self._active or self._storage_fault is not None:
            try:
                self._event(
                    "body_lost",
                    "runtime_safety",
                    f"body-lost:{uuid.uuid4()}",
                    {"reason": reason},
                )
            except sqlite3.Error as error:
                # A failed durable safety record must never prevent stopping
                # the body. This process cannot enable play again afterward.
                self._storage_fault = type(error).__name__
            finally:
                try:
                    await asyncio.to_thread(self._safe_stop)
                except (OSError, ConnectionError, TimeoutError, RuntimeError):
                    pass

    async def watchdog(self) -> None:
        while True:
            try:
                if self._storage_fault is None:
                    if self.runtime.state.play_enabled:
                        await asyncio.to_thread(self._body_preflight)
                    self.close_expired()
            except asyncio.CancelledError:
                raise
            except sqlite3.Error as error:
                self._storage_fault = type(error).__name__
                await self._latch_body_lost("storage_failure")
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                ValueError,
            ) as error:
                await self._latch_body_lost(type(error).__name__)
            await asyncio.sleep(2.0)

    def _close_episode(
        self,
        trace_id: str,
        events: list[Event],
        *,
        verdict: OutcomeVerdict,
        metrics: dict[str, float] | None = None,
        reason: str | None = None,
    ) -> tuple[Episode, Outcome]:
        episode_id = f"episode:{trace_id}"
        invitation = next(event for event in events if event.kind == "invitation")
        episode = Episode(
            episode_id=episode_id,
            trace_id=trace_id,
            started_at=invitation.occurred_at,
            ended_at=events[-1].occurred_at,
            event_ids=tuple(event.event_id for event in events),
            intent_ids=(f"intent:{trace_id}",),
            outcome_ids=(f"outcome:{trace_id}",),
            runtime_versions={
                "duckmind_rule": "preference-v1",
                "service": "first-board-v1",
            },
        )
        outcome = Outcome(
            outcome_id=f"outcome:{trace_id}",
            episode_id=episode_id,
            verdict=verdict,
            evaluator_version="owner-feedback-v1" if metrics else "execution-v1",
            evidence_event_ids=episode.event_ids,
            metrics=metrics or {},
            reason=reason,
        )
        self.runtime.store.commit_episode(episode, outcome)
        return episode, outcome

    def close_expired(self) -> int:
        """Reconcile a crash window; close no-feedback games as UNKNOWN."""

        # The first pass rebuilds unfinished work after a crash. Subsequent
        # watchdog ticks read only new events while pending traces stay here.
        for entry in self.runtime.store.list_events(after_seq=self._reconcile_seq):
            self._reconcile_seq = entry.seq
            if entry.event.kind in {
                "invitation",
                "behavior_decision",
                "skill_completed",
                "skill_execution",
                "explicit_feedback",
            }:
                self._reconcile_pending[entry.event.trace_id].append(entry.event)
        closed = 0
        now = _now()
        for trace_id, events in tuple(self._reconcile_pending.items()):
            invitations = [event for event in events if event.kind == "invitation"]
            completions = [event for event in events if event.kind == "skill_completed"]
            feedback = [event for event in events if event.kind == "explicit_feedback"]
            already_closed = self.runtime.store.has_episode(f"episode:{trace_id}")
            score = feedback[0].payload.get("score") if len(feedback) == 1 else None
            actor = invitations[0].payload.get("actor") if len(invitations) == 1 else None
            eligible = (
                len(invitations) == 1
                and len(completions) == 1
                and len(feedback) == 1
                and isinstance(actor, str)
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and score in (0.0, 1.0)
                and invitations[0].source.kind == "owner_request"
                and invitations[0].source.reliability >= 0.8
                and invitations[0].payload.get("activity") == "look_game"
                and completions[0].source.kind == "skill_layer"
                and completions[0].source.reliability >= 0.8
                and completions[0].payload.get("status") == "completed"
                and completions[0].payload.get("activity") == "look_game"
                and feedback[0].source.kind == "owner_feedback"
                and feedback[0].source.reliability >= 0.8
                and feedback[0].payload.get("actor") == actor
                and feedback[0].payload.get("activity") == "look_game"
                and timedelta(0)
                <= feedback[0].occurred_at - completions[0].occurred_at
                <= FEEDBACK_WINDOW
            )
            if eligible:
                assert isinstance(actor, str) and isinstance(score, (int, float))
                verdict = OutcomeVerdict.SUCCESS if score == 1.0 else OutcomeVerdict.FAILURE
                stored = self.runtime.store.outcome_by_id(f"outcome:{trace_id}")
                if stored is not None and (
                    stored.verdict is not verdict
                    or stored.evidence_event_ids != tuple(event.event_id for event in events)
                ):
                    del self._reconcile_pending[trace_id]
                    continue  # a closed UNKNOWN/other outcome can never be reclassified
                if not already_closed:
                    self._close_episode(
                        trace_id,
                        events,
                        verdict=verdict,
                        metrics={"feedback": float(score)},
                    )
                    closed += 1
                    already_closed = True
                if not self.runtime.store.has_learning_for_episode(
                    f"episode:{trace_id}", f"preferences.{actor}.look_game"
                ):
                    episode, outcome = self._close_episode(
                        trace_id,
                        events,
                        verdict=verdict,
                        metrics={"feedback": float(score)},
                    )
                    proposal = self._learner.propose(
                        episode=episode,
                        outcome=outcome,
                        evidence=events,
                        state=self.runtime.state,
                    )
                    if proposal is not None:
                        self._learner.commit(proposal, episode=episode, outcome=outcome, at=now)
                        self.runtime.refresh()
                del self._reconcile_pending[trace_id]
                continue
            if already_closed:
                del self._reconcile_pending[trace_id]
                continue
            if now - events[-1].occurred_at > FEEDBACK_WINDOW:
                if len(invitations) != 1:
                    # An orphan/ambiguous trace is not a valid episode.
                    del self._reconcile_pending[trace_id]
                    continue
                self._close_episode(
                    trace_id,
                    events,
                    verdict=OutcomeVerdict.UNKNOWN,
                    reason="feedback_window_expired",
                )
                closed += 1
                del self._reconcile_pending[trace_id]
        return closed

    async def handle(self, request: dict[str, Any]) -> dict[str, object]:
        action = request.get("action")
        if action == "invite" or action == "enable":
            if self._storage_fault is not None:
                raise RuntimeError("storage_fault_requires_restart")
            if self._stops_in_flight:
                raise ValueError("stop_in_progress")
            if self._active:
                raise ValueError("interaction_active")
        if action == "invite":
            # Reserve admission before preflight yields. Otherwise multiple
            # socket clients can all pass the idle check before one dispatches.
            self._active = True
            try:
                return await self._handle_with_storage_guard(request)
            finally:
                self._active = False
        if action == "stop":
            self._stops_in_flight += 1
            try:
                return await self._handle_with_storage_guard(request)
            finally:
                self._stops_in_flight -= 1
        return await self._handle_with_storage_guard(request)

    async def _handle_with_storage_guard(self, request: dict[str, Any]) -> dict[str, object]:
        try:
            return await self._handle(request)
        except sqlite3.Error as error:
            self._storage_fault = type(error).__name__
            await self._latch_body_lost("storage_failure")
            raise

    async def _handle(self, request: dict[str, Any]) -> dict[str, object]:
        action = request.get("action")
        request_id = request.get("request_id")
        if not isinstance(action, str) or not isinstance(request_id, str):
            raise ValueError("action and request_id are required")
        try:
            uuid.UUID(request_id)
        except ValueError as error:
            raise ValueError("request_id must be a UUID") from error
        if action == "status":
            approval = self._head_approval_state()
            return {
                **self.runtime.health(),
                "backend": self.backend,
                "active_interaction": self._active,
                "body_socket_present": self.robotd_socket.exists(),
                "preferences": dict(self.runtime.state.preferences),
                "learning_updates": self.runtime.store.count("learning_updates"),
                "play_enabled": self.runtime.state.play_enabled
                and self._storage_fault is None
                and (self.backend != "real" or approval == "approved"),
                "real_head_profile_approval": approval,
                "storage_fault": self._storage_fault,
            }
        if action == "stop":
            self._cancel.set()
            self._stop_epoch += 1
            stop_event_id = f"stop:{request_id}"
            previous_stop = self.runtime.store.event_by_id(stop_event_id)
            if previous_stop is None:
                self._event("owner_stop", "owner_control", stop_event_id, {})
            elif self.runtime.state.play_enabled:
                # A retry is still a stop if an enable occurred after the
                # original request. Preserve that event and append evidence
                # of the new safety transition before yielding to transport.
                self._event(
                    "owner_stop",
                    "owner_control",
                    f"stop-reassert:{uuid.uuid4()}",
                    {"request_id": request_id},
                    trace_id=previous_stop.trace_id,
                    parent_event_ids=(stop_event_id,),
                )
            stop_worker = asyncio.create_task(asyncio.to_thread(self._safe_stop))
            try:
                stopped = await asyncio.shield(stop_worker)
            except asyncio.CancelledError:
                # Cancellation cannot release stop admission while the thread
                # still owns or is waiting for the physical command lock.
                await asyncio.gather(stop_worker, return_exceptions=True)
                raise
            except (OSError, ConnectionError, TimeoutError, RuntimeError):
                stopped = False
            return {"play_enabled": False, "robotd_stop_accepted": stopped}
        if action == "enable":
            if self._active:
                raise ValueError("interaction_active")
            if self.runtime.store.event_by_id(f"enable:{request_id}") is not None:
                raise ValueError("duplicate_request_id")
            epoch = self._stop_epoch
            battery, policy, fallen, limp = await asyncio.to_thread(self._body_preflight)
            if epoch != self._stop_epoch:
                raise ValueError("stopped_during_preflight")
            self._event(
                "robot_state",
                "robotd_state",
                f"state:{request_id}",
                {"battery_fraction": battery, "fallen": fallen, "limp": limp},
            )
            self._event("owner_play_enabled", "owner_control", f"enable:{request_id}", {})
            self._cancel.clear()
            return {"play_enabled": self.runtime.state.play_enabled, "policy": policy}
        if action == "invite":
            if not self.runtime.state.play_enabled:
                raise ValueError("play_disabled")
            actor = request.get("actor")
            if not isinstance(actor, str) or not NAME.fullmatch(actor):
                raise ValueError("actor must be a short lowercase ID")
            if request.get("activity") != "look_game":
                raise ValueError("only look_game is enabled in the first-board slice")
            trace_id = request_id
            if self.runtime.store.events_for_trace(trace_id):
                raise ValueError("duplicate_request_id")
            epoch = self._stop_epoch
            try:
                battery, _, fallen, limp = await asyncio.to_thread(self._body_preflight)
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                ValueError,
            ) as error:
                await self._latch_body_lost(type(error).__name__)
                raise RuntimeError("body_preflight_failed") from error
            if epoch != self._stop_epoch or not self.runtime.state.play_enabled:
                raise ValueError("stopped_during_preflight")
            self._event(
                "robot_state",
                "robotd_state",
                f"state:{request_id}",
                {"battery_fraction": battery, "fallen": fallen, "limp": limp},
            )
            invitation = self._event(
                "invitation",
                "owner_request",
                f"invitation:{trace_id}",
                {"actor": actor, "activity": "look_game"},
                trace_id=trace_id,
            )
            decision = choose_invitation(self.runtime.state, invitation)
            decision_event = self._event(
                "behavior_decision",
                "system1",
                f"decision:{trace_id}",
                {
                    "goal": f"play:{actor}:look_game"
                    if decision.choice is BehaviorChoice.ENGAGE
                    else None,
                    "choice": decision.choice.value,
                    "score": decision.score,
                    "reason": decision.reason,
                    "invitation_event_id": invitation.event_id,
                    "influences": sorted(decision.influences),
                },
                trace_id=trace_id,
            )
            if decision.choice is not BehaviorChoice.ENGAGE:
                self._close_episode(
                    trace_id,
                    [invitation, decision_event],
                    verdict=OutcomeVerdict.UNKNOWN,
                    reason=f"decision:{decision.choice.value}",
                )
                return {
                    "interaction_id": trace_id,
                    "choice": decision.choice.value,
                    "outcome": "unknown",
                }
            self._cancel.clear()
            gaze_worker = asyncio.create_task(asyncio.to_thread(self._run_gaze, trace_id))
            try:
                status = await asyncio.shield(gaze_worker)
            except asyncio.CancelledError:
                # asyncio cancellation does not stop a worker thread. Keep
                # admission reserved until it exits, and revoke play now.
                try:
                    await self._latch_body_lost("interaction_cancelled")
                finally:
                    await asyncio.gather(gaze_worker, return_exceptions=True)
                raise
            except (
                OSError,
                ConnectionError,
                TimeoutError,
                RuntimeError,
                ValueError,
            ) as error:
                status = SkillStatus(trace_id, SkillPhase.INTERRUPTED, 0.0, type(error).__name__)
            if self._cancel.is_set() and status.phase is SkillPhase.COMPLETED:
                status = SkillStatus(
                    status.execution_id, SkillPhase.INTERRUPTED, status.at, "owner_stop"
                )
            if status.phase is not SkillPhase.COMPLETED and status.reason != "owner_stop":
                await self._latch_body_lost(status.reason or "interaction_not_completed")
            kind = "skill_completed" if status.phase is SkillPhase.COMPLETED else "skill_execution"
            execution = self._event(
                kind,
                "skill_layer",
                f"execution:{trace_id}",
                {
                    "activity": "look_game",
                    "intent_id": f"intent:{trace_id}",
                    "execution_id": status.execution_id,
                    "status": status.phase.value,
                    "reason": status.reason,
                },
                trace_id=trace_id,
            )
            if status.phase is not SkillPhase.COMPLETED:
                self._close_episode(
                    trace_id,
                    [invitation, decision_event, execution],
                    verdict=OutcomeVerdict.INTERRUPTED
                    if status.phase is SkillPhase.INTERRUPTED
                    else OutcomeVerdict.UNKNOWN,
                    reason=status.reason or status.phase.value,
                )
            return {
                "interaction_id": trace_id,
                "choice": decision.choice.value,
                "execution": status.phase.value,
                "execution_reason": status.reason,
                "feedback_allowed": status.phase is SkillPhase.COMPLETED,
            }
        if action == "feedback":
            interaction_id = request.get("interaction_id")
            score = request.get("score")
            if (
                not isinstance(interaction_id, str)
                or not isinstance(score, int)
                or isinstance(score, bool)
                or score not in (0, 1)
            ):
                raise ValueError("feedback needs interaction_id and binary score")
            if self.runtime.store.event_by_id(f"feedback:{request_id}") is not None:
                raise ValueError("duplicate_request_id")
            if self.runtime.store.has_episode(f"episode:{interaction_id}"):
                raise ValueError("interaction_already_closed")
            events = self.runtime.store.events_for_trace(interaction_id)
            invitation_events = [event for event in events if event.kind == "invitation"]
            completions = [event for event in events if event.kind == "skill_completed"]
            if (
                len(invitation_events) != 1
                or len(completions) != 1
                or completions[0].payload.get("status") != "completed"
            ):
                raise ValueError("interaction_not_completed")
            if _now() - completions[0].occurred_at > FEEDBACK_WINDOW:
                self._close_episode(
                    interaction_id,
                    events,
                    verdict=OutcomeVerdict.UNKNOWN,
                    reason="feedback_window_expired",
                )
                raise ValueError("feedback_window_expired")
            actor = invitation_events[0].payload.get("actor")
            if not isinstance(actor, str):
                raise ValueError("interaction_actor_malformed")
            feedback = self._event(
                "explicit_feedback",
                "owner_feedback",
                f"feedback:{request_id}",
                {"actor": actor, "activity": "look_game", "score": float(score)},
                trace_id=interaction_id,
            )
            evidence = [*events, feedback]
            verdict = OutcomeVerdict.SUCCESS if score else OutcomeVerdict.FAILURE
            episode, outcome = self._close_episode(
                interaction_id, evidence, verdict=verdict, metrics={"feedback": float(score)}
            )
            proposal = self._learner.propose(
                episode=episode,
                outcome=outcome,
                evidence=evidence,
                state=self.runtime.state,
            )
            if proposal is not None:
                self._learner.commit(proposal, episode=episode, outcome=outcome, at=_now())
                self.runtime.refresh()
            return {
                "interaction_id": interaction_id,
                "outcome": verdict.value,
                "learning_update_id": proposal.update_id if proposal else None,
            }
        raise ValueError("unsupported_action")


async def serve(controller: PetController, socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if socket_path.exists():
        if not socket_path.is_socket():
            raise RuntimeError("refusing to replace a non-socket path")
        try:
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(0.2)
                probe.connect(str(socket_path))
        except (ConnectionRefusedError, FileNotFoundError):
            socket_path.unlink()
        else:
            raise RuntimeError("another Duck Mind service owns the socket")

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not raw:
                writer.close()  # readiness probes may connect and close
                return
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("request_too_large")
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request_must_be_object")
            result = await controller.handle(request)
            response: dict[str, object] = {"ok": True, "result": result}
        except (
            ValueError,
            RuntimeError,
            OSError,
            ConnectionError,
            TimeoutError,
            json.JSONDecodeError,
            sqlite3.Error,
        ) as error:
            response = {"ok": False, "error": str(error)[:200]}
        try:
            writer.write(
                json.dumps(response, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            )
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = await asyncio.start_unix_server(
        handle_client, path=str(socket_path), limit=MAX_REQUEST_BYTES
    )
    os.chmod(socket_path, 0o600)
    monitor = asyncio.create_task(controller.watchdog())
    serving = asyncio.create_task(server.serve_forever())
    try:
        async with server:
            done, _ = await asyncio.wait((monitor, serving), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise RuntimeError("Duck Mind service or watchdog exited unexpectedly")
    finally:
        monitor.cancel()
        serving.cancel()
        await asyncio.gather(monitor, serving, return_exceptions=True)
        await controller._latch_body_lost("service_shutdown")
        if socket_path.is_socket():
            socket_path.unlink()


async def _serve_until_signal(controller: PetController, socket_path: Path) -> None:
    """Turn systemd SIGTERM into orderly server cancellation and body stop."""

    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    registered: list[signal.Signals] = []
    tasks: list[asyncio.Task[object]] = []
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, shutdown.set)
            registered.append(signum)
        service_task = asyncio.create_task(serve(controller, socket_path))
        signal_task = asyncio.create_task(shutdown.wait())
        tasks.extend((service_task, signal_task))
        done, _ = await asyncio.wait(
            (service_task, signal_task), return_when=asyncio.FIRST_COMPLETED
        )
        if service_task in done:
            await service_task
            raise RuntimeError("Duck Mind service exited without a shutdown signal")
        service_task.cancel()
        outcome = (await asyncio.gather(service_task, return_exceptions=True))[0]
        if not isinstance(outcome, asyncio.CancelledError):
            if isinstance(outcome, BaseException):
                raise outcome
            raise RuntimeError("Duck Mind service did not cancel on shutdown")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for signum in registered:
            loop.remove_signal_handler(signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, default=Path("/var/lib/duckmind/duckmind.sqlite3"))
    parser.add_argument("--socket", type=Path, default=Path("/run/duckmind/duckmind.sock"))
    parser.add_argument("--robotd", type=Path, default=Path("/run/robotd.sock"))
    parser.add_argument("--backend", choices=("real", "simulator"), default="real")
    parser.add_argument("--identity", default="duck")
    parser.add_argument(
        "--sim-body-trace",
        type=Path,
        help="single-interaction head-game body trace; simulator backend only",
    )
    parser.add_argument(
        "--approve-real-v5-head-profile",
        action="store_true",
        help="operator acceptance after supervised head-motion bench test; never set on first boot",
    )
    parser.add_argument(
        "--approved-release-id",
        help="release ID accepted during the supervised bench test; required with real approval",
    )
    args = parser.parse_args()
    with DuckRuntime(args.state, identity=args.identity, born_at=_now()) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=args.robotd,
            backend=args.backend,
            approve_real_v5_head_profile=args.approve_real_v5_head_profile,
            approved_release_id=args.approved_release_id,
            sim_body_trace=args.sim_body_trace,
        )
        asyncio.run(_serve_until_signal(controller, args.socket))


if __name__ == "__main__":
    main()
