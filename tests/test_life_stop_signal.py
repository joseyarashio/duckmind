"""A clean stop through the real path writes the mind (the 2026-10-10 report: a SIGTERM at 21:05
wrote core.json but no neural.json): the signal goes to the server's PID, the way stop-8791.sh sends
it (to the PID that listens, with a wrapper process above it), twice as the script does, and the
snapshot and Duck Core's save are on disk when it has stopped."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CHILD = Path(__file__).with_name("life_stop_child.py")


def run_and_stop(folder: Path, *, second_term: bool, wrapper: bool) -> tuple[int, str]:
    command = [sys.executable, str(CHILD), str(folder)]
    if wrapper:
        # (a parent above it, as uv is above the server: the stop script TERMs the listening PID)
        command = ["sh", "-c", 'python_cmd="$0"; "$python_cmd" "$@"', *command]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert proc.stdout is not None
    line = ""
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.startswith("READY"):
            break
    assert line.startswith("READY"), (line, proc.stderr.read() if proc.stderr else "")
    pid = int(line.split()[1])
    time.sleep(0.3)
    os.kill(pid, signal.SIGTERM)
    if second_term:
        time.sleep(0.01)
        os.kill(pid, signal.SIGTERM)  # (stop-8791.sh sends it again to a server still stopping)
    out, err = proc.communicate(timeout=60)
    return proc.returncode, out + err


def test_a_sigterm_to_the_servers_pid_writes_the_mind_and_duck_cores_save(tmp_path) -> None:
    folder = tmp_path / "duck"
    code, said = run_and_stop(folder, second_term=False, wrapper=False)
    assert code == 0 and "STOPPED" in said, said
    neural = json.loads((folder / "neural.json").read_text())
    assert neural["why"] == "stop" and len(neural["learner"]["people"]) == 1
    core = json.loads((folder / "core.json").read_text())
    assert core["bridge"] == {"api": "bridge-v1"}


def test_a_second_sigterm_while_it_is_stopping_does_not_cut_the_saving_short_and_a_wrapper_is_no_matter(
    tmp_path,
) -> None:
    folder = tmp_path / "duck"
    code, said = run_and_stop(folder, second_term=True, wrapper=True)
    assert code == 0 and "STOPPED" in said, said
    assert json.loads((folder / "neural.json").read_text())["why"] == "stop"
