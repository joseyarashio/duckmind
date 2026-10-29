"""Real local process supervision tests; no training stack, GPU or network."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from duckmind import training


def _alive(pid: int) -> bool:
    # An orphan killed on Linux can briefly be a zombie awaiting its init reaper.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform == "linux":
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False
    return True


def _wait_pid(path: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.01)
    pytest.fail(f"child did not become ready: {path}")


def _assert_stopped(pid: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.01)
    assert not _alive(pid), f"owned child {pid} is still running"


def _kill_owned_group(log: Path) -> None:
    if log.exists() and log.read_text().strip():
        group = int(log.read_text().splitlines()[0])
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("leader_exits", [False, True], ids=["timeout", "successful-leader"])
def test_phase_cleans_term_ignoring_descendant_after_leader_exits(tmp_path, leader_exits):
    child_pid_path = tmp_path / "child.pid"
    log = tmp_path / "phase.log"
    child = (
        "import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader = (
        "import os,subprocess,sys,time; from pathlib import Path\n"
        "print(os.getpid(),flush=True)\n"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])\n"
        "while not Path(sys.argv[2]).exists(): time.sleep(.01)\n"
        "if sys.argv[3] == 'wait': time.sleep(30)\n"
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        result = training.run_phase(
            [
                sys.executable,
                "-c",
                leader,
                child,
                str(child_pid_path),
                "exit" if leader_exits else "wait",
            ],
            cwd=tmp_path,
            env=dict(os.environ),
            log=log,
            timeout_s=1,
        )
        assert result.timed_out is not leader_exits
        if leader_exits:
            assert result.returncode == 0
        _assert_stopped(_wait_pid(child_pid_path))
        assert unrelated.poll() is None, "cleanup must only target its own new process group"
    finally:
        _kill_owned_group(log)
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_preflight_receives_remaining_job_budget(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.setattr(training, "verify_upstream", lambda _: {"commit": "test-only"})
    received = []

    def probe(*_, timeout_s=60, **__):
        received.append(timeout_s)
        return {}

    monkeypatch.setattr(training, "probe_environment", probe)
    result = training.run_job(
        upstream,
        Path(sys.executable),
        tmp_path / "job",
        training.TrainingSpec(timeout_s=1),
        execute=True,
    )
    assert result["status"] == "blocked_preflight"
    assert len(received) == 1 and 0 < received[0] <= 1


def test_preflight_timeout_reaps_probe_and_preserves_json_channel(tmp_path, monkeypatch):
    pid_path = tmp_path / "probe.pid"
    monkeypatch.setattr(
        training,
        "ENVIRONMENT_PROBE",
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)",
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            training.probe_environment(Path(sys.executable), tmp_path, timeout_s=0.5)
        _assert_stopped(_wait_pid(pid_path))
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    monkeypatch.setattr(
        training,
        "ENVIRONMENT_PROBE",
        "import sys; print('diagnostic',file=sys.stderr); print('{\"probe\": true}')",
    )
    assert training.probe_environment(Path(sys.executable), tmp_path, timeout_s=3) == {
        "probe": True
    }


def test_sigterm_during_preflight_reaps_probe_and_records_cancelled_job(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    job = tmp_path / "job"
    probe_pid_path = tmp_path / "probe.pid"
    script = (
        "import os,sys; from pathlib import Path; from duckmind import training as t\n"
        "t.verify_upstream = lambda _: {'commit': 'test-only'}\n"
        "t.ENVIRONMENT_PROBE = 'import os,time; from pathlib import Path; Path(' + "
        "repr(sys.argv[3]) + ').write_text(str(os.getpid())); time.sleep(30)'\n"
        "t.run_job(Path(sys.argv[1]),Path(sys.executable),Path(sys.argv[2]),"
        "t.TrainingSpec(timeout_s=30),execute=True)\n"
    )
    owner = subprocess.Popen(
        [sys.executable, "-c", script, str(upstream), str(job), str(probe_pid_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe_pid = None
    try:
        probe_pid = _wait_pid(probe_pid_path)
        owner.terminate()
        owner.wait(timeout=8)
        _assert_stopped(probe_pid)
        report = json.loads((job / "job.json").read_text())
        assert report["status"] == "cancelled"
        assert report["training_gate"] == "not_run"
        assert report["physical_hardware_gate"] == "not_run"
        assert report["finished_at"]
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=3)
        if probe_pid is not None:
            try:
                os.kill(probe_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigterm_during_success_cleanup_is_deferred_not_lost(tmp_path):
    pid_path, term_path = tmp_path / "child.pid", tmp_path / "term.seen"
    log = tmp_path / "phase.log"
    child = (
        "import os,signal,sys,time; from pathlib import Path\n"
        "signal.signal(signal.SIGTERM,lambda *_: Path(sys.argv[2]).touch())\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    leader = (
        "import os,subprocess,sys,time; from pathlib import Path\n"
        "print(os.getpid(),flush=True)\n"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]])\n"
        "while not Path(sys.argv[2]).exists(): time.sleep(.01)\n"
    )
    owner_script = (
        "import os,sys; from pathlib import Path; from duckmind.training import run_phase\n"
        "try:\n"
        " run_phase([sys.executable,'-c',sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]],"
        "cwd=Path(sys.argv[1]),env=dict(os.environ),log=Path(sys.argv[2]),timeout_s=10)\n"
        "except KeyboardInterrupt: sys.exit(42)\n"
    )
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            owner_script,
            str(tmp_path),
            str(log),
            leader,
            child,
            str(pid_path),
            str(term_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        child_pid = _wait_pid(pid_path)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not term_path.exists():
            time.sleep(0.01)
        assert term_path.exists(), "leader must have exited and cleanup sent TERM"
        owner.terminate()
        owner.wait(timeout=8)
        assert owner.returncode == 42, "cancellation during cleanup must not become success"
        _assert_stopped(child_pid)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=3)
        _kill_owned_group(log)
