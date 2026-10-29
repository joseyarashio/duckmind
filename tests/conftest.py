"""Shared test settings: the phase-0 render check.

The camera model (P5), the light meter, the duck's-eye camera, its JPEG encoder and the
MuJoCo comparison of the two collision models need numpy, MuJoCo 3.10 (and Pillow for one
decode check) and the official checkout. The standard check runs without them, so those tests
skip there; the run then says how many and how to run them (the render check in
docs/validation/run-duck-simulator.md). With ``DUCKMIND_REQUIRE_RENDER=1`` — as the render check
sets it — any of them that skips fails instead, so the render check can never pass by skipping.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

import duckmind.world.legacy  # noqa: F401  (registers LegacyWorld: v4_1 ducks live in it)
from duckmind.duckcore import engine as duck_engine
from duckmind.world.legacy import LegacyWorld

RENDER_FILES = frozenset(
    {
        "test_camera_model.py",
        "test_eye_light.py",
        "test_eyecam.py",
        "test_bodyhost.py",
        "test_simhome.py",  # phase 2: the world server's in-process MuJoCo cases
        "test_render_scene.py",  # phase 2: the eye in a home (lights per room, the ladder)
        "test_homelayout.py",  # phase 2: every home compiles, spawns clear, page == physics
        "test_world_nestpad.py",  # phase 2: the nest's contact, pin and tags in a home
        "test_world_dock.py",  # the charging dock (2026-09-27): both styles, contacts, PG, tag
        "test_world_hand.py",  # phase 2: the hand carries and rights the physics duck
        "test_world_people.py",  # phase 2: family bodies walk over, 40 cm from the duck
        "test_world_battery.py",  # phase 2: the world battery on the nest's contact
        "test_world_props.py",  # phase 2: things placed, tidied, rolled; impacts
        "test_world_kit.py",  # phase 2: the household inside the official body server
        "test_media_frame_protocol.py",  # phase 2 P1: media.frame against upstream mediad
        "test_tof_op_protocol.py",  # phase 2 P1: the tof op, SimDepth and the real tofd
        "test_tof_model.py",  # phase 2: the world's multi-ray ToF
        "test_eye_render.py",  # phase 2: the camera's capture branch
        "test_sensors.py",  # phase 2: the sensors extension in the world server
        "test_reprojector_vectors.py",  # phase 2 P4: the Reprojector port against upstream
        # phase 3: duck-eye (numpy; renders through the world camera; pupil-apriltags; onnx)
        "test_eye_camera.py",
        "test_eye_frames.py",
        "test_eye_server.py",
        "test_eye_floorline.py",
        "test_eye_floorline_scenes.py",
        "test_eye_nesttag.py",
        "test_eye_things.py",
        "test_eye_detect.py",
        "test_nest_contract.py",  # the dock tag from duck-eye's read to duck-body's pose
        # phase 3: the gate harness (headless robotd path, the lockstep rig, upstream vectors)
        "test_headless_duck.py",
        "test_rig_determinism.py",
        "test_headless_vectors.py",
        "test_wall_walk_replay.py",
        "test_uyvy_vectors.py",
        # phase 3: curiosity's provisional jaw (A4); 「用眼睛做」 on the page (A6)
        "test_world_jaw.py",
        "test_simweb_eyedemo.py",
        # live voice LV0: a rendered duck-eye frame handed to the live connection's frame source
        "test_live_eye_frames.py",
    }
)
# (the live smoke run stays opt-in in both checks; so do the tests of the official tofd,
# which needs a local cargo build)
OPT_IN = ("DUCKMIND_LIVE_SMOKE", "DUCKMIND_TOFD")
REQUIRE = "DUCKMIND_REQUIRE_RENDER"
ONLY = "DUCKMIND_RENDER_ONLY"  # =1: run the RENDER_FILES only (the render check's own list)

MACHINE_VERIFY = Path(__file__).resolve().parents[1] / "tools" / "machine_verify"


class _MachineVerifyAbsent(pytest.Module):
    """A test_machine_verify_* file in a checkout without tools/machine_verify (the public
    repository): reported as skipped instead of failing to import its tools."""

    def collect(self) -> Any:
        pytest.skip("tools/machine_verify (the internal machine verification) is not here")


def pytest_pycollect_makemodule(module_path: Path, parent: Any) -> Any:
    if module_path.name.startswith("test_machine_verify_") and not MACHINE_VERIFY.is_dir():
        return _MachineVerifyAbsent.from_parent(parent, path=module_path)
    return None


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "population_defaults: keep the population runs' percept sampling (design A13) for "
        "this test instead of checking every percept",
    )


@pytest.fixture(autouse=True)
def _check_every_percept(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase-4 design A13: a population run trusts an in-process world after each duck's
    first 1,000 percepts; the tests check every percept a world sends, on both sides of the
    port (Duck Core's ``verify_left`` and LegacyWorld's own check)."""
    if request.node.get_closest_marker("population_defaults") is None:
        monkeypatch.setattr(duck_engine, "VERIFY_PORT_PERCEPTS", None)
        monkeypatch.setattr(LegacyWorld, "VALIDATE_FIRST", None)


def _reason(report: Any) -> str:
    longrepr = report.longrepr
    return str(longrepr[2]) if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)


def _render_skip(report: Any) -> bool:
    if not report.skipped:
        return False
    if str(report.nodeid).split("::")[0].rsplit("/", 1)[-1] not in RENDER_FILES:
        return False
    return not any(flag in _reason(report) for flag in OPT_IN)


def pytest_collection_modifyitems(config: Any, items: list[pytest.Item]) -> None:
    if os.environ.get(ONLY) != "1":
        return
    keep = [item for item in items if item.path.name in RENDER_FILES]
    dropped = [item for item in items if item.path.name not in RENDER_FILES]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: Any) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if os.environ.get(REQUIRE) == "1" and _render_skip(report):
        reason = _reason(report)
        report.outcome = "failed"
        report.longrepr = f"{REQUIRE}=1: this render test must run here, not skip ({reason})"


def pytest_terminal_summary(terminalreporter: Any) -> None:
    skipped = [r for r in terminalreporter.stats.get("skipped", []) if _render_skip(r)]
    if skipped:
        terminalreporter.write_line(
            f"phase-0 render tests skipped here: {len(skipped)} (camera model P5, light meter, "
            "duck's-eye camera and JPEG, collision models; phase 2–3 homes, sensors, duck-eye, "
            "the gate harness). They need numpy + MuJoCo 3.10 + Pillow (+ pupil-apriltags, "
            "onnxruntime, onnx, opencv-python-headless): run the render check in "
            "docs/validation/run-duck-simulator.md."
        )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live voice (LV1): a test that uses this can not reach the network or the Keychain: a
    connect, a name lookup or a ``security`` call fails the test on the spot."""
    import socket
    import subprocess

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this live-voice test tried to reach the network")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    real_run = subprocess.run

    def guarded_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if cmd and "security" in str(cmd[0] if isinstance(cmd, (list, tuple)) else cmd):
            raise AssertionError("this live-voice test tried to read the Keychain")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
