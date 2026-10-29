import asyncio
import hashlib
import json
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from duckmind.behavior import BehaviorChoice, choose_invitation
from duckmind.body import BodyAck, RobotObservation, SafetyObservation
from duckmind.cli import request as cli_request
from duckmind.pet_body_trace import replay_front_gaze_trace
from duckmind.policy_approval import official_v5_walk_file_matches
from duckmind.provenance import Event, OutcomeVerdict, SourceRef
from duckmind.runtime import DuckRuntime
from duckmind.service import PetController, serve
from duckmind.tmpdir import short_tmp

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


class FakeSafety:
    def __init__(self, *, battery=75.0, healthy=True, stopped=None, policy_report=None):
        self.battery = battery
        self.healthy = healthy
        self.stopped = stopped
        self.policy_report = policy_report

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def health(self):
        return {
            "healthy": self.healthy,
            "battery": None if self.battery is None else {"percent": self.battery, "volts": 7.5},
        }

    def stop(self):
        if self.stopped is not None:
            self.stopped.append(True)
        return BodyAck(True)

    def policies(self):
        if self.policy_report is not None:
            return self.policy_report
        return {
            "mode": "walk",
            "enabled": True,
            "homed": True,
            "change_error": None,
            "slots": [
                {
                    "slot": "walk",
                    "path": "/opt/robot/policies/current/velstand.onnx",
                    "origin": "official",
                    "overridden": False,
                    "error": None,
                }
            ],
        }


class FakeBody:
    def __init__(self, *, release=None):
        self.release = release
        self.dispatched = []
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def dispatch_intent(self, intent):
        self.dispatched.append(intent)
        return BodyAck(True)

    def observe(self):
        self.calls += 1
        if self.release is not None and self.calls >= 3:
            assert self.release.wait(2.0)
        return RobotObservation(
            monotonic_s=time.monotonic(),
            policy="walk",
            movement_requested=(0.0, 0.0, 0.0),
            movement_applied=(0.0, 0.0, 0.0),
            limited_by=(),
            safety=SafetyObservation(False, False),
            backend="simulator",
            head_tracking_error_rad=0.02,
        )

    def stop(self):
        return BodyAck(True)


def command(action, **extra):
    return {"action": action, "request_id": str(uuid.uuid4()), **extra}


def release_record(tmp_path: Path, release_id: str = "0.1.0-rc6") -> Path:
    record = tmp_path / "RELEASE_ID"
    record.write_text(f"{release_id}\n")
    return record


def actor_that_explores(state):
    for index in range(1000):
        actor = f"person{index}"
        invitation = Event(
            event_id=f"trial-{index}",
            kind="invitation",
            occurred_at=NOW,
            observed_at=NOW,
            source=SourceRef("owner_request", "test", 1.0),
            trace_id="trial",
            payload={"actor": actor, "activity": "look_game"},
        )
        if choose_invitation(state, invitation).choice is BehaviorChoice.ENGAGE:
            return actor
    raise AssertionError("no deterministic exploration case found")


@pytest.mark.parametrize("score,expected_outcome", [(1, "success"), (0, "failure")])
def test_completed_local_game_can_train_and_restart_is_passive(
    tmp_path, monkeypatch, score, expected_outcome
) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    path = tmp_path / "pet.sqlite3"
    body = FakeBody()
    with DuckRuntime(path, identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(),
        )

        async def run():
            assert (await controller.handle(command("enable")))["play_enabled"] is True
            actor = actor_that_explores(runtime.state)
            invitation = await controller.handle(
                command("invite", actor=actor, activity="look_game")
            )
            assert invitation["execution"] == "completed"
            assert invitation["feedback_allowed"] is True
            outcome = await controller.handle(
                command("feedback", interaction_id=invitation["interaction_id"], score=score)
            )
            assert outcome["outcome"] == expected_outcome
            assert outcome["learning_update_id"] is not None
            with pytest.raises(ValueError, match="interaction_already_closed"):
                await controller.handle(
                    command("feedback", interaction_id=invitation["interaction_id"], score=score)
                )
            return actor

        actor = asyncio.run(run())
        preference = runtime.state.preferences[f"preferences.{actor}.look_game"]
        assert 0.5 < preference <= 0.505 if score else 0.495 <= preference < 0.5
        assert len(body.dispatched) == 1
        assert body.dispatched[0].method == "robot.head"
    with DuckRuntime(path, identity="duck", born_at=NOW) as runtime:
        PetController(runtime, robotd_socket=tmp_path / "robot.sock", backend="simulator")
        assert runtime.state.play_enabled is False
        assert runtime.store.count("learning_updates") == 1


def test_simulator_service_records_replayable_head_game(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    trace_path = tmp_path / "look-game.body.json"

    class TraceBody(FakeBody):
        def observe(self):
            observed = super().observe()
            return replace(observed, robot_time_s=float(self.calls))

    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=TraceBody,
            safety_factory=FakeSafety,
            sim_body_trace=trace_path,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            return await controller.handle(command("invite", actor=actor, activity="look_game"))

        interaction = asyncio.run(run())
        assert interaction["execution"] == "completed"
        assert trace_path.is_file()
        assert json.loads(trace_path.read_text())["interaction_id"] == interaction["interaction_id"]
        assert replay_front_gaze_trace(trace_path).passed


def test_body_trace_cannot_be_enabled_for_real_robot(tmp_path) -> None:
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        with pytest.raises(ValueError, match="simulator only"):
            PetController(
                runtime,
                robotd_socket=tmp_path / "robot.sock",
                backend="real",
                sim_body_trace=tmp_path / "forbidden.json",
            )


def test_real_board_requires_known_battery_before_enabling(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=lambda: FakeSafety(battery=None),
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=release_record(tmp_path),
        )
        with pytest.raises(RuntimeError, match="battery_unknown"):
            asyncio.run(controller.handle(command("enable")))
        assert runtime.state.play_enabled is False


def test_real_board_rejects_malformed_battery_before_enabling(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=lambda: FakeSafety(battery=True),
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=release_record(tmp_path),
        )
        with pytest.raises(RuntimeError, match="battery_malformed"):
            controller._body_preflight()
        assert runtime.state.play_enabled is False


@pytest.mark.parametrize(
    "unsafe_state,reason",
    [
        ("other_policy", "unexpected_policy:kick_left"),
        ("requested_motion", "motion_command_active"),
        ("applied_motion", "motion_command_active"),
        ("nan_motion", "motion_command_active"),
    ],
)
def test_head_game_preflight_refuses_other_policy_or_motion(
    tmp_path, monkeypatch, unsafe_state, reason
) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)

    class UnsafeBody(FakeBody):
        def observe(self):
            state = super().observe()
            if unsafe_state == "other_policy":
                return replace(state, policy="kick_left")
            if unsafe_state == "requested_motion":
                return replace(state, movement_requested=(0.2, 0.0, 0.0))
            if unsafe_state == "nan_motion":
                return replace(state, movement_requested=(float("nan"), 0.0, 0.0))
            return replace(state, movement_applied=(0.2, 0.0, 0.0))

    body = UnsafeBody()
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=FakeSafety,
        )
        with pytest.raises(RuntimeError, match=reason):
            asyncio.run(controller.handle(command("enable")))
        assert runtime.state.play_enabled is False
        assert body.dispatched == []


def test_real_motion_profile_requires_supervised_approval(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )
        with pytest.raises(RuntimeError, match="not_approved"):
            asyncio.run(controller.handle(command("enable")))
        assert runtime.state.play_enabled is False


def test_real_approval_flag_alone_cannot_unlock_new_release(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    record = release_record(tmp_path)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
            approve_real_v5_head_profile=True,
            release_id_path=record,
        )
        assert controller._head_approval_state() == "release_not_bound"
        with pytest.raises(RuntimeError, match="release_not_bound"):
            controller._body_preflight()
        controller.approved_release_id = "0.1.0-rc5"
        assert controller._head_approval_state() == "release_mismatch"
        status = asyncio.run(controller.handle(command("status")))
        assert status["real_head_profile_approval"] == "release_mismatch"
        assert status["play_enabled"] is False
        with pytest.raises(RuntimeError, match="release_mismatch"):
            asyncio.run(controller.handle(command("enable")))
        assert runtime.state.play_enabled is False


def test_real_approval_revokes_if_selected_release_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    record = release_record(tmp_path)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=record,
        )
        assert controller._head_approval_state() == "approved"
        record.write_text("0.1.0-rc7\n")
        assert controller._head_approval_state() == "release_mismatch"
        with pytest.raises(RuntimeError, match="release_mismatch"):
            controller._body_preflight()
        assert runtime.state.play_enabled is False


def test_real_preflight_rejects_loaded_same_basename_from_other_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    report = FakeSafety().policies()
    report["slots"][0]["path"] = "/var/lib/robot/policies/velstand.onnx"
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=lambda: FakeSafety(policy_report=report),
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=release_record(tmp_path),
        )
        with pytest.raises(RuntimeError, match="loaded_v5_walk_mismatch"):
            asyncio.run(controller.handle(command("enable")))
        assert runtime.state.play_enabled is False


def test_real_preflight_rejects_mismatched_official_walk_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=release_record(tmp_path),
            policy_file_check=lambda: False,
        )
        with pytest.raises(RuntimeError, match="official_v5_walk_bytes_mismatch"):
            asyncio.run(controller.handle(command("enable")))
        assert runtime.state.play_enabled is False


def test_real_watchdog_revokes_play_after_official_walk_file_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    model = tmp_path / "velstand.onnx"
    model.write_bytes(b"reviewed-v5-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=lambda: FakeSafety(stopped=stopped),
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=release_record(tmp_path),
            policy_file_check=lambda: official_v5_walk_file_matches(model, expected_sha256=digest),
        )

        async def run() -> None:
            assert (await controller.handle(command("enable")))["play_enabled"] is True
            model.write_bytes(b"different-model")
            monitor = asyncio.create_task(controller.watchdog())
            try:
                for _ in range(100):
                    if not runtime.state.play_enabled:
                        break
                    await asyncio.sleep(0.01)
                assert runtime.state.play_enabled is False
            finally:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)

        asyncio.run(run())
        assert stopped == [True]


def test_real_watchdog_revokes_play_after_policy_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    safety = FakeSafety(stopped=[])
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=lambda: safety,
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=release_record(tmp_path),
            policy_file_check=lambda: True,
        )

        async def run() -> None:
            assert (await controller.handle(command("enable")))["play_enabled"] is True
            report = safety.policies()
            report["slots"][0]["overridden"] = True
            safety.policy_report = report
            monitor = asyncio.create_task(controller.watchdog())
            try:
                for _ in range(100):
                    if not runtime.state.play_enabled:
                        break
                    await asyncio.sleep(0.01)
                assert runtime.state.play_enabled is False
            finally:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)

        asyncio.run(run())
        assert safety.stopped == [True]


def test_watchdog_latches_play_off_after_release_switch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    record = release_record(tmp_path)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=record,
            policy_file_check=lambda: True,
        )

        async def run() -> None:
            assert (await controller.handle(command("enable")))["play_enabled"] is True
            record.write_text("0.1.0-rc7\n")
            monitor = asyncio.create_task(controller.watchdog())
            try:
                for _ in range(100):
                    if not runtime.state.play_enabled:
                        break
                    await asyncio.sleep(0.01)
                assert runtime.state.play_enabled is False
            finally:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)

        asyncio.run(run())


def test_watchdog_revokes_play_when_another_policy_takes_over(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)

    class SwitchingBody(FakeBody):
        policy = "walk"

        def observe(self):
            return replace(super().observe(), policy=self.policy)

    body = SwitchingBody()
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run() -> None:
            assert (await controller.handle(command("enable")))["play_enabled"] is True
            body.policy = "kick_left"
            monitor = asyncio.create_task(controller.watchdog())
            try:
                for _ in range(100):
                    if not runtime.state.play_enabled:
                        break
                    await asyncio.sleep(0.01)
                assert body.calls >= 2
                assert runtime.state.play_enabled is False
            finally:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)

        asyncio.run(run())
        assert stopped == [True]


@pytest.mark.parametrize("fault", ["missing", "linked", "malformed"])
def test_real_approval_requires_regular_release_record(tmp_path, monkeypatch, fault) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    record = release_record(tmp_path)
    if fault == "missing":
        record.unlink()
    elif fault == "linked":
        record.unlink()
        target = tmp_path / "other-release-id"
        target.write_text("0.1.0-rc6\n")
        record.symlink_to(target)
    else:
        record.write_text("0.1.0-rc6\nextra\n")
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="real",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
            approve_real_v5_head_profile=True,
            approved_release_id="0.1.0-rc6",
            release_id_path=record,
        )
        assert controller._head_approval_state() != "approved"
        with pytest.raises(RuntimeError, match="real_v5_head_profile_"):
            controller._body_preflight()


def test_no_feedback_closes_unknown_and_never_learns(tmp_path, monkeypatch) -> None:
    clock = [NOW]
    monkeypatch.setattr("duckmind.service._now", lambda: clock[0])
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            return await controller.handle(command("invite", actor=actor, activity="look_game"))

        result = asyncio.run(run())
        assert result["execution"] == "completed"
        assert runtime.store.count("episodes") == 0
        assert controller.close_expired() == 0
        clock[0] = NOW.replace(minute=3)
        assert controller.close_expired() == 1
        assert runtime.store.count("outcomes") == 1
        assert runtime.store.count("learning_updates") == 0
        with pytest.raises(ValueError, match="interaction_already_closed"):
            asyncio.run(
                controller.handle(
                    command("feedback", interaction_id=result["interaction_id"], score=1)
                )
            )


def test_reconciliation_reuses_pending_traces_and_reads_only_new_events(
    tmp_path, monkeypatch
) -> None:
    clock = [NOW]
    monkeypatch.setattr("duckmind.service._now", lambda: clock[0])
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            return actor, await controller.handle(
                command("invite", actor=actor, activity="look_game")
            )

        actor, invitation = asyncio.run(run())
        assert invitation["execution"] == "completed"
        original = runtime.store.list_events
        cursors = []

        def recording_list_events(*, after_seq=0):
            cursors.append(after_seq)
            return original(after_seq=after_seq)

        monkeypatch.setattr(runtime.store, "list_events", recording_list_events)
        assert controller.close_expired() == 0
        first_cursor = runtime.store.latest_event_seq()
        assert cursors == [0]
        clock[0] = NOW + timedelta(seconds=30)
        assert controller.close_expired() == 0
        assert cursors == [0, first_cursor]

        controller._event(
            "explicit_feedback",
            "owner_feedback",
            f"feedback:{uuid.uuid4()}",
            {"actor": actor, "activity": "look_game", "score": 1.0},
            trace_id=invitation["interaction_id"],
        )
        before_feedback_scan = runtime.store.latest_event_seq() - 1
        calls_before_reconcile = len(cursors)
        assert controller.close_expired() == 1
        assert cursors[calls_before_reconcile] == before_feedback_scan
        assert runtime.store.count("learning_updates") == 1
        assert controller.close_expired() == 0
        assert controller.close_expired() == 0
        assert cursors[-1] == runtime.store.latest_event_seq()


def test_reconciliation_after_restart_finishes_closed_episode_learning(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    path = tmp_path / "pet.sqlite3"
    with DuckRuntime(path, identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            result = await controller.handle(command("invite", actor=actor, activity="look_game"))
            return actor, result["interaction_id"]

        actor, trace_id = asyncio.run(run())
        controller._event(
            "explicit_feedback",
            "owner_feedback",
            f"feedback:{uuid.uuid4()}",
            {"actor": actor, "activity": "look_game", "score": 1.0},
            trace_id=trace_id,
        )
        controller._close_episode(
            trace_id,
            runtime.store.events_for_trace(trace_id),
            verdict=OutcomeVerdict.SUCCESS,
            metrics={"feedback": 1.0},
        )
        assert runtime.store.count("learning_updates") == 0

    with DuckRuntime(path, identity="duck", born_at=NOW) as restarted:
        controller = PetController(
            restarted, robotd_socket=tmp_path / "robot.sock", backend="simulator"
        )
        assert controller.close_expired() == 0
        assert restarted.store.count("learning_updates") == 1
        assert controller.close_expired() == 0


def test_crash_between_feedback_and_episode_is_reconciled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            result = await controller.handle(command("invite", actor=actor, activity="look_game"))
            return actor, result["interaction_id"]

        actor, trace_id = asyncio.run(run())
        controller._event(
            "explicit_feedback",
            "owner_feedback",
            f"feedback:{uuid.uuid4()}",
            {"actor": actor, "activity": "look_game", "score": 1.0},
            trace_id=trace_id,
        )
        assert controller.close_expired() == 1
        assert runtime.store.count("episodes") == 1
        assert runtime.store.count("learning_updates") == 1
        assert controller.close_expired() == 0


def test_late_feedback_stays_unknown_across_repeated_reconciliation(tmp_path, monkeypatch) -> None:
    clock = [NOW]
    monkeypatch.setattr("duckmind.service._now", lambda: clock[0])
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            result = await controller.handle(command("invite", actor=actor, activity="look_game"))
            return actor, result["interaction_id"]

        actor, trace_id = asyncio.run(run())
        clock[0] = NOW + timedelta(seconds=121)
        controller._event(
            "explicit_feedback",
            "owner_feedback",
            f"feedback:{uuid.uuid4()}",
            {"actor": actor, "activity": "look_game", "score": 1.0},
            trace_id=trace_id,
        )
        clock[0] = NOW + timedelta(minutes=5)
        assert controller.close_expired() == 1
        assert controller.close_expired() == 0
        assert runtime.store.outcome_by_id(f"outcome:{trace_id}").verdict is OutcomeVerdict.UNKNOWN
        assert runtime.store.count("learning_updates") == 0


def test_crash_after_episode_before_learning_is_reconciled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            result = await controller.handle(command("invite", actor=actor, activity="look_game"))
            return actor, result["interaction_id"]

        actor, trace_id = asyncio.run(run())
        controller._event(
            "explicit_feedback",
            "owner_feedback",
            f"feedback:{uuid.uuid4()}",
            {"actor": actor, "activity": "look_game", "score": 1.0},
            trace_id=trace_id,
        )
        controller._close_episode(
            trace_id,
            runtime.store.events_for_trace(trace_id),
            verdict=OutcomeVerdict.SUCCESS,
            metrics={"feedback": 1.0},
        )
        assert runtime.store.count("learning_updates") == 0
        assert controller.close_expired() == 0
        assert runtime.store.count("learning_updates") == 1
        assert controller.close_expired() == 0


def test_stop_interrupts_gaze_without_waiting_for_body_stream(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    release = threading.Event()
    body = FakeBody(release=release)
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            invite = asyncio.create_task(
                controller.handle(command("invite", actor=actor, activity="look_game"))
            )
            for _ in range(100):
                if body.calls >= 3:
                    break
                await asyncio.sleep(0.01)
            assert body.calls >= 3
            stop = await controller.handle(command("stop"))
            assert stop["robotd_stop_accepted"] is True
            release.set()
            result = await invite
            assert result["execution"] == "interrupted"
            assert result["feedback_allowed"] is False

        asyncio.run(run())
        assert stopped == [True]
        assert runtime.state.play_enabled is False
        assert runtime.store.count("learning_updates") == 0


@pytest.mark.parametrize("fault", ["fall", "other_policy", "requested_motion"])
def test_gaze_interruption_immediately_revokes_play_without_watchdog(
    tmp_path, monkeypatch, fault
) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)

    class InterruptedBody(FakeBody):
        def observe(self):
            state = super().observe()
            if self.calls < 3:
                return state
            if fault == "fall":
                return replace(state, safety=SafetyObservation(True, False))
            if fault == "other_policy":
                return replace(state, policy="kick_left")
            return replace(state, movement_requested=(0.2, 0.0, 0.0))

    body = InterruptedBody()
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            result = await controller.handle(command("invite", actor=actor, activity="look_game"))
            assert result["execution"] == "interrupted"
            assert result["feedback_allowed"] is False

        asyncio.run(run())
        assert runtime.state.play_enabled is False
        assert stopped == [True]
        assert runtime.store.count("learning_updates") == 0


def test_refused_head_command_revokes_play_before_another_invitation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)

    class RefusingBody(FakeBody):
        def dispatch_intent(self, intent):
            self.dispatched.append(intent)
            return BodyAck(False, "head_refused")

    body = RefusingBody()
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            result = await controller.handle(command("invite", actor=actor, activity="look_game"))
            assert result["execution"] == "refused"
            assert result["feedback_allowed"] is False
            with pytest.raises(ValueError, match="play_disabled"):
                await controller.handle(command("invite", actor=actor, activity="look_game"))

        asyncio.run(run())
        assert runtime.state.play_enabled is False
        assert stopped == [True]
        assert runtime.store.count("learning_updates") == 0


def test_concurrent_invites_reserve_admission_before_preflight(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    entered = threading.Event()
    release = threading.Event()
    body = FakeBody()
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=FakeSafety,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)

            def preflight():
                entered.set()
                assert release.wait(2.0)
                return 0.75, "walk", False, False

            monkeypatch.setattr(controller, "_body_preflight", preflight)
            invitation = asyncio.create_task(
                controller.handle(command("invite", actor=actor, activity="look_game"))
            )
            assert await asyncio.to_thread(entered.wait, 2.0)
            try:
                for _ in range(2):
                    with pytest.raises(ValueError, match="interaction_active"):
                        await controller.handle(
                            command("invite", actor=actor, activity="look_game")
                        )
                with pytest.raises(ValueError, match="interaction_active"):
                    await controller.handle(command("enable"))
            finally:
                release.set()
            assert (await invitation)["execution"] == "completed"
            assert controller._active is False

        asyncio.run(run())
        assert len(body.dispatched) == 1


def test_stop_latches_before_transport_and_blocks_new_play(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    entered = threading.Event()
    release = threading.Event()
    body = FakeBody()

    class SlowStop(FakeSafety):
        def stop(self):
            entered.set()
            assert release.wait(2.0)
            return BodyAck(True)

    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=SlowStop,
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            stop = asyncio.create_task(controller.handle(command("stop")))
            assert await asyncio.to_thread(entered.wait, 2.0)
            try:
                assert runtime.state.play_enabled is False
                assert controller._cancel.is_set()
                with pytest.raises(ValueError, match="stop_in_progress"):
                    await controller.handle(command("invite", actor=actor, activity="look_game"))
                with pytest.raises(ValueError, match="stop_in_progress"):
                    await controller.handle(command("enable"))
            finally:
                release.set()
            assert (await stop)["robotd_stop_accepted"] is True
            assert controller._stops_in_flight == 0

        asyncio.run(run())
        assert body.dispatched == []


def test_retried_stop_reasserts_disabled_play_after_intervening_enable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    path = tmp_path / "pet.sqlite3"
    body = FakeBody()
    stopped = []
    with DuckRuntime(path, identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            stop_request = command("stop")
            stop_event_id = f"stop:{stop_request['request_id']}"
            await controller.handle(stop_request)
            original_stop = runtime.store.event_by_id(stop_event_id)
            assert original_stop is not None
            await controller.handle(stop_request)
            assert (
                sum(entry.event.kind == "owner_stop" for entry in runtime.store.list_events()) == 1
            )
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            assert runtime.state.play_enabled is True
            result = await controller.handle(stop_request)
            assert result["play_enabled"] is False
            assert result["robotd_stop_accepted"] is True
            assert runtime.state.play_enabled is False
            with pytest.raises(ValueError, match="play_disabled"):
                await controller.handle(command("invite", actor=actor, activity="look_game"))
            assert runtime.store.event_by_id(stop_event_id) == original_stop
            stops = [
                entry.event
                for entry in runtime.store.list_events()
                if entry.event.kind == "owner_stop"
            ]
            assert len(stops) == 2
            assert stops[-1].event_id != stop_event_id
            assert stops[-1].parent_event_ids == (stop_event_id,)
            assert stops[-1].trace_id == original_stop.trace_id
            assert stops[-1].payload["request_id"] == stop_request["request_id"]

        asyncio.run(run())
        assert stopped == [True, True, True]
        assert body.dispatched == []
    with DuckRuntime(path, identity="duck", born_at=NOW) as restarted:
        assert restarted.state.play_enabled is False


def test_cancelled_invite_keeps_admission_until_worker_exits(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    release = threading.Event()
    body = FakeBody(release=release)
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=lambda: body,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            await controller.handle(command("enable"))
            actor = actor_that_explores(runtime.state)
            invitation = asyncio.create_task(
                controller.handle(command("invite", actor=actor, activity="look_game"))
            )
            for _ in range(100):
                if body.calls >= 3:
                    break
                await asyncio.sleep(0.01)
            assert body.calls >= 3
            invitation.cancel()
            try:
                for _ in range(100):
                    if stopped:
                        break
                    await asyncio.sleep(0.01)
                assert stopped == [True]
                assert runtime.state.play_enabled is False
                assert controller._active is True
                assert not invitation.done()
                with pytest.raises(ValueError, match="interaction_active"):
                    await controller.handle(command("enable"))
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await invitation
            assert controller._active is False

        asyncio.run(run())
        assert len(body.dispatched) == 1
        assert not any(
            entry.event.kind == "skill_completed" for entry in runtime.store.list_events()
        )
        assert runtime.store.count("learning_updates") == 0


def test_storage_failure_cannot_prevent_stop_or_reenable_play(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            await controller.handle(command("enable"))

            def disk_full(*_args, **_kwargs):
                raise sqlite3.OperationalError("database or disk is full")

            monkeypatch.setattr(runtime, "ingest", disk_full)
            with pytest.raises(sqlite3.OperationalError, match="disk is full"):
                await controller.handle(command("stop"))
            assert stopped == [True]
            assert controller._cancel.is_set()
            status = await controller.handle(command("status"))
            assert status["play_enabled"] is False
            assert status["storage_fault"] == "OperationalError"
            with pytest.raises(RuntimeError, match="storage_fault_requires_restart"):
                await controller.handle(command("enable"))

        asyncio.run(run())


def test_watchdog_database_error_latches_play_and_keeps_monitor_alive(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    stopped = []
    with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
        controller = PetController(
            runtime,
            robotd_socket=tmp_path / "robot.sock",
            backend="simulator",
            body_factory=FakeBody,
            safety_factory=lambda: FakeSafety(stopped=stopped),
        )

        async def run():
            await controller.handle(command("enable"))

            def broken_database():
                raise sqlite3.OperationalError("disk I/O error")

            monkeypatch.setattr(controller, "close_expired", broken_database)
            monitor = asyncio.create_task(controller.watchdog())
            try:
                for _ in range(100):
                    if stopped:
                        break
                    await asyncio.sleep(0.01)
                assert stopped == [True]
                assert not monitor.done()
                assert runtime.state.play_enabled is False
                with pytest.raises(RuntimeError, match="storage_fault_requires_restart"):
                    await controller.handle(command("enable"))
            finally:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)

        asyncio.run(run())


def test_local_socket_is_owner_only_and_cli_can_read_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    with tempfile.TemporaryDirectory(prefix="dm-service-", dir=short_tmp()) as directory:
        socket_path = Path(directory) / "pet.sock"
        with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
            controller = PetController(
                runtime, robotd_socket=tmp_path / "robot.sock", backend="simulator"
            )

            async def run():
                task = asyncio.create_task(serve(controller, socket_path))
                for _ in range(100):
                    if socket_path.is_socket():
                        break
                    await asyncio.sleep(0.01)
                assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
                response = await asyncio.to_thread(cli_request, socket_path, command("status"))
                assert response["ok"] is True
                assert response["result"]["play_enabled"] is False
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            asyncio.run(run())
            assert not socket_path.exists()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_shutdown_signal_gracefully_closes_service_socket(signum: signal.Signals) -> None:
    with tempfile.TemporaryDirectory(prefix="dm-signal-", dir="/tmp") as directory:
        root = Path(directory)
        socket_path = root / "duckmind.sock"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "duckmind.service",
                "--state",
                str(root / "state.sqlite3"),
                "--socket",
                str(socket_path),
                "--robotd",
                str(root / "missing-robotd.sock"),
                "--backend",
                "simulator",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 5
            while not socket_path.is_socket() and process.poll() is None:
                if time.monotonic() >= deadline:
                    raise AssertionError("Duck Mind service did not create its socket")
                time.sleep(0.02)
            assert socket_path.is_socket()
            process.send_signal(signum)
            _, stderr = process.communicate(timeout=5)
            assert process.returncode == 0, stderr.decode()
            assert not socket_path.exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_unexpected_watchdog_exit_stops_service_and_body(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("duckmind.service._now", lambda: NOW)
    stopped = []
    with tempfile.TemporaryDirectory(prefix="dm-monitor-", dir=short_tmp()) as directory:
        socket_path = Path(directory) / "pet.sock"
        with DuckRuntime(tmp_path / "pet.sqlite3", identity="duck", born_at=NOW) as runtime:
            controller = PetController(
                runtime,
                robotd_socket=tmp_path / "robot.sock",
                backend="simulator",
                body_factory=FakeBody,
                safety_factory=lambda: FakeSafety(stopped=stopped),
            )

            async def broken_monitor():
                raise RuntimeError("unexpected monitor failure")

            monkeypatch.setattr(controller, "watchdog", broken_monitor)

            async def run():
                await controller.handle(command("enable"))
                with pytest.raises(RuntimeError, match="unexpected monitor failure"):
                    await serve(controller, socket_path)

            asyncio.run(run())
            assert stopped == [True]
            assert runtime.state.play_enabled is False
            assert not socket_path.exists()
