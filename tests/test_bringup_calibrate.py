"""The bench tool that measures a real duck's body calibration (python -m duckmind.bringup, docs/deploy/bring-up.md):
the guided recording through the staged poses, the fit of the leg-load bands and of the ToF touch band, the replay
check, the report, and the document the brain then reads — end to end on a fake duck whose legs draw another scale
than the simulator's, and over the real link code against a fake robotd."""

from __future__ import annotations

import json
import os
import random
import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_duckbody_depth import SENSOR_POS, cast, sensor
from test_real_duck_body import TOKEN, StateRobotd

from duckmind.bringup import __main__ as cli
from duckmind.bringup import fit as F
from duckmind.bringup import record as R
from duckmind.bringup import tofcal as T
from duckmind.duckbody import calibration as C
from duckmind.duckbody.proprio import ProprioConfig
from duckmind.remotebody.bridge import Bridge, Channel
from duckmind.tmpdir import short_tmp

WALK, SEAT = "a" * 64, "b" * 64
INFO = {
    "backend": "real",
    "policies": {"set": "v8", "walk_sha256": WALK, "sitstand_sha256": SEAT},
    "robotd_config": {"board": "face-board-v1"},
}
DT = 0.02
# a body's leg load by what it is doing, in the simulator's units (proprio's module doc: ≈ 9 standing in the hand,
# ≈ 13 seated in the hand, ≈ 35 standing and ≈ 25 seated on the floor, ≈ 110 walking, ≈ 130 stepping in the air)
LOADS = {
    ("air", "stand"): (9.0, 1.2),
    ("air", "sit"): (13.0, 1.2),
    ("floor", "stand"): (35.0, 2.0),
    ("floor", "sit"): (25.0, 1.2),
    ("floor", "walk"): (110.0, 12.0),
    ("air", "walk"): (130.0, 12.0),
    ("air", "flail"): (125.0, 8.0),
}


class FakeDuck:
    """A duck on the bench: its posture follows ``sit_toggle`` after a second, its legs load by where it is (``where``:
    the person's doing) on its own ``scale``; time is the ticks read (20 ms each), nothing sleeps."""

    def __init__(self, scale: float = 12.0, *, seed: int = 5, loads: dict | None = None) -> None:
        self.info = dict(INFO)
        self.scale = scale
        self.loads = {**LOADS, **(loads or {})}
        self.random = random.Random(seed)
        self.t = 100.0
        self.posture = "sit"
        self.where = "floor"
        self.where_since = self.t
        self.flailing = False
        self.toggle_at: float | None = None
        self.move_at = -1.0
        self.vx = 0.0
        self.fall_at: float | None = None
        self.sent: list[tuple[float, str, float]] = []
        self.board_mm: int | None = None
        self.tof_reads: Any = (
            None  # (true mm) → (distance_mm, status) of 64 zones; None: the geometry's
        )
        self.picked_up = False

    # ── the person ──
    def put(self, where: str) -> None:
        if where != self.where:
            self.where, self.where_since = where, self.t
            self.flailing = False

    # ── Body ──
    def sit_toggle(self) -> None:
        self.sent.append((self.t, "sit_toggle", 0.0))
        self.toggle_at = self.t + 1.0

    def move(self, vx: float) -> None:
        self.sent.append((self.t, "move", vx))
        self.vx, self.move_at = vx, self.t

    def stop(self) -> None:
        self.sent.append((self.t, "stop", 0.0))
        self.vx = 0.0

    def read_state(self, timeout_s: float) -> dict[str, Any]:
        self.t += DT
        if self.toggle_at is not None and self.t >= self.toggle_at:
            self.posture = "stand" if self.posture == "sit" else "sit"
            self.toggle_at = None
        walking = self.vx >= 0.25 and self.t - self.move_at < 0.5 and self.posture == "stand"
        if walking and self.where == "air":
            self.flailing = True
        doing = (
            "walk"
            if walking
            else ("flail" if self.flailing and self.posture == "stand" else self.posture)
        )
        mean, spread = self.loads[(self.where, doing)]
        target = mean + self.random.uniform(-spread, spread)
        # (a hand takes 0.6 s to take the weight off, or to give it back)
        other = "air" if self.where == "floor" else "floor"
        blend = min(1.0, (self.t - self.where_since) / 0.6)
        before = self.loads[(other, self.posture)][0]
        load = (before + (target - before) * blend) * self.scale
        currents = [0.0] * 15
        for i in (0, 1, 2, 3, 4, 10, 11, 12, 13, 14):
            currents[i] = load / 10.0
        rock = (0.6 if self.where == "floor" else 0.04) if walking else 0.02
        fallen = self.fall_at is not None and self.t >= self.fall_at
        if fallen:
            self.fall_at = None  # (one tick of it: the person stands it up again)
        q = sensor(0).quat
        return {
            "t_ns": int(round(self.t * 1e9)),
            "policy": "held"
            if self.picked_up and self.where == "air"
            else ("walk" if walking else self.posture),
            "move": {"requested": [self.vx if walking else 0.0, 0, 0], "applied": [0, 0, 0]},
            "safety": {
                "fallen": fallen,
                "limp": False,
                "picked_up": self.picked_up and self.where == "air",
                "gravity": [0.0, 0.0, -1.0],
            },
            "joints": [0.0] * 15,
            "currents_ma": currents,
            "imu": {
                "gyro": [rock * self.random.choice((-1, 1)), 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
            "odom": {"position": [0.0, 0.0, 0.125], "yaw": 0.0},
            "frames": {"tof": {"pos": list(SENSOR_POS), "quat": [q.w, q.x, q.y, q.z]}},
        }

    def read_tof(self, timeout_s: float) -> dict[str, Any] | None:
        if self.board_mm is None:
            return None
        if self.tof_reads is not None:
            mm, st = self.tof_reads(self.board_mm)
        else:
            mm, st = cast(sensor(0), wall_x=SENSOR_POS[0] + max(self.board_mm, 5) / 1000.0)
        return {"t_ns": int(round(self.t * 1e9)), "distance_mm": mm, "status": st}


class Person:
    """The person at the bench, doing what the prompts say (and, in a passive recording, working the duck's own
    control: ``own_control``)."""

    def __init__(self, duck: FakeDuck, *, own_control: bool = False) -> None:
        self.duck, self.own_control = duck, own_control
        self.said: list[str] = []

    def ready(self, segment: R.Segment, rep: int) -> None:
        duck = self.duck
        duck.put("air" if segment.name.startswith("air_") else "floor")
        duck.board_mm = segment.tof_mm
        if self.own_control and segment.posture is not None and duck.posture != segment.posture:
            duck.posture = segment.posture
        if self.own_control and segment.walk is not None:
            # (the duck's own gamepad: forward for four seconds, a second after the recording starts)
            duck.vx, duck.move_at = 0.30, duck.t + R.SETTLE_S + 1.0 + 4.0 - 0.5
            duck.t_walk = (duck.t + R.SETTLE_S + 1.0, duck.t + R.SETTLE_S + 5.0)

    def cue(self, segment: R.Segment, rep: int, name: str) -> None:
        if name in ("lift", "set_down"):
            self.duck.put("air" if name == "lift" else "floor")
        elif self.own_control:
            self.duck.posture = name  # ("sit" / "stand": the duck's own control)

    def note(self, text: str) -> None:
        self.said.append(text)


def session(
    tmp_path: Path, duck: FakeDuck, *, stages=(1, 2, 3), tof: bool = True, drive: bool = True
):
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=tof, unit="bench duck"))
    person = Person(duck)
    try:
        done = R.record(duck, R.plan_for(stages, tof=tof), sink, person, drive=drive)
    finally:
        sink.close()
    return done, person


# ── the whole way: record → fit → the document the brain reads ──


def test_the_bench_session_gives_that_bodys_own_bands_and_the_brain_finds_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    duck = FakeDuck(scale=12.0)
    done, _ = session(tmp_path, duck)
    assert set(done) == {s.name for s in R.PLAN} | {f"tof_{mm}" for mm in R.TOF_DISTANCES_MM}
    assert cli.fit(tmp_path, save=True) == 0
    found = C.find("face-board-v1", WALK, SEAT)
    assert found.calibration is not None and found.calibration.unit == "bench duck"
    bands = found.calibration.proprio
    # between this body's hand (≈ 9 × 12) and its floor (≈ 35 × 12): nowhere near the simulator's 20
    assert 12 * 11 < bands.lift_load < bands.unsure_load <= bands.floor_band[0] < 12 * 34
    assert (
        12 * 15 < bands.lift_load_sit < bands.unsure_load_sit <= bands.floor_band_sit[0] < 12 * 25
    )
    assert bands.unsure_load_reverse < bands.lift_load
    # the flailing level and the band's high edge: above the floor's loads (≈ 35), under the air's (≈ 125)
    assert 12 * 45 < bands.flail_load < bands.floor_band[1] < 12 * 125
    assert (
        found.calibration.tof == C.SIMULATOR.tof
    )  # (this sensor reads as the simulator's: its numbers fit)
    data = json.loads((tmp_path / cli.REPORT_JSON).read_text(encoding="utf-8"))
    assert data["why_not"] == [] and data["legs"]["replay"]["failures"] == []
    assert data["legs"]["replay"]["lifts"] == {
        "lift_stand": {"claimed": 3, "of": 3},
        "lift_sit": {"claimed": 3, "of": 3},
    }
    assert data["legs"]["replay"]["claim_after_cue_s"]["max"] < 2.0
    report = (tmp_path / cli.REPORT_MD).read_text(encoding="utf-8")
    assert "PASS" in report and "in the hand, standing" in report and "ToF touch band" in report
    # what load cannot tell apart is said, not hidden
    assert any("stepping in the air" in text for text in data["legs"]["cannot_separate"])
    assert data["legs"]["replay"]["cycles"] == 5 and "sit-stand cycles" in report
    assert "installed" in capsys.readouterr().out


def test_the_bands_follow_the_bodys_own_scale(tmp_path: Path) -> None:
    small, large = tmp_path / "a", tmp_path / "b"
    session(small, FakeDuck(scale=1.0))
    session(large, FakeDuck(scale=40.0))
    one = F.fit(R.read(small / R.RECORDING)[1]).bands
    forty = F.fit(R.read(large / R.RECORDING)[1]).bands
    assert one is not None and forty is not None
    assert forty.lift_load == pytest.approx(40 * one.lift_load, rel=0.02)
    assert forty.floor_band[0] == pytest.approx(40 * one.floor_band[0], rel=0.02)
    # on the simulator's own scale the fit lands by the simulator's hand-set numbers
    sim = C.SIMULATOR.proprio
    assert one.lift_load == pytest.approx(sim.lift_load, abs=4.0)
    assert one.unsure_load == pytest.approx(sim.unsure_load, abs=4.0)
    assert one.floor_band[0] == pytest.approx(sim.floor_band[0], abs=4.0)
    assert one.lift_load_sit == pytest.approx(sim.lift_load_sit, abs=3.0)


def test_a_seat_that_loads_the_legs_no_more_than_a_hand_turns_the_seated_claim_off(
    tmp_path: Path,
) -> None:
    session(tmp_path, FakeDuck(loads={("floor", "sit"): (14.0, 1.2)}))
    header, lines = R.read(tmp_path / R.RECORDING)
    legs = F.fit(lines)
    assert legs.bands is not None and legs.usable
    assert (legs.bands.lift_load_sit, legs.bands.unsure_load_sit) == (0.0, 0.0)
    assert any("seated" in text for text in legs.cannot)
    cal = cli.build(header, legs, T.fit(lines), "0" * 64)
    assert cal is not None and C.parse(cal.to_json()) == cal


def test_a_body_whose_hand_and_floor_loads_overlap_gets_no_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    session(tmp_path, FakeDuck(loads={("air", "stand"): (30.0, 2.0)}))
    assert cli.fit(tmp_path, save=True) == 1
    assert C.find("face-board-v1", WALK, SEAT).why == C.WHY_NONE  # nothing was installed
    report = (tmp_path / cli.REPORT_MD).read_text(encoding="utf-8")
    assert "NOT YET" in report and "cannot be told from standing on the floor" in report


def test_without_the_walk_stage_there_is_no_calibration_yet(tmp_path: Path) -> None:
    done, _ = session(tmp_path, FakeDuck(), stages=(1, 2))
    assert "walk" not in done
    header, lines = R.read(tmp_path / R.RECORDING)
    why = cli.why_not(header, F.fit(lines), T.fit(lines))
    assert why == ["the segment walk was not recorded"]
    # the stages are recorded one visit at a time into the same folder
    duck = FakeDuck()
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=True, unit=""))
    R.record(duck, R.plan_for((3,), tof=True), sink, Person(duck), drive=True)
    sink.close()
    header, lines = R.read(tmp_path / R.RECORDING)
    assert cli.why_not(header, F.fit(lines), T.fit(lines)) == []


# ── what the tool sends ──


def test_a_passive_recording_sends_nothing_but_the_final_stop(tmp_path: Path) -> None:
    duck = FakeDuck()
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=False, unit=""))
    plan = [s for s in R.PLAN if s.walk is None]
    R.record(duck, plan, sink, Person(duck, own_control=True), drive=False)
    sink.close()
    assert [(what, vx) for _, what, vx in duck.sent] == [("move", 0.0), ("stop", 0.0)]


def test_a_driven_recording_sends_only_the_posture_toggle_and_the_short_slow_walk(
    tmp_path: Path,
) -> None:
    duck = FakeDuck()
    session(tmp_path, duck, tof=False)
    assert {what for _, what, _ in duck.sent} == {"sit_toggle", "move", "stop"}
    walking = [(t, vx) for t, what, vx in duck.sent if what == "move" and vx != 0.0]
    assert {vx for _, vx in walking} == {R.WALK_VX}
    bouts: list[list[float]] = []
    for t, _ in walking:
        if bouts and t - bouts[-1][-1] < 0.1:
            bouts[-1].append(t)
        else:
            bouts.append([t])
    assert len(bouts) == 1 + 3  # the legs in the air once, three walks on the floor
    toggles = sum(what == "sit_toggle" for _, what, _ in duck.sent)
    assert 10 <= toggles <= 16  # five sit-stand cycles, and the posture each pose needs
    assert all(bout[-1] - bout[0] <= 4.0 + DT for bout in bouts)
    assert duck.sent[-1][1] == "stop" and duck.sent[-2][1:] == ("move", 0.0)


def test_a_fall_voids_the_repetition_and_it_is_recorded_again(tmp_path: Path) -> None:
    duck = FakeDuck()
    walk = next(s for s in R.PLAN if s.name == "walk")
    duck.posture = "stand"
    duck.fall_at = duck.t + R.SETTLE_S + 3.0
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=False, unit=""))
    person = Person(duck)
    done = R.record(duck, [walk], sink, person, drive=True)
    sink.close()
    assert any("does not count" in text and "fell" in text for text in person.said)
    _, lines = R.read(tmp_path / R.RECORDING)
    reps = F.reps_of(lines)
    assert [r.rep for r in reps] == [0, 1, 2] and done["walk"] == sum(len(r.ticks) for r in reps)
    assert not any(t.state.fallen for r in reps for t in r.ticks)
    # … and every way out ends with the stop
    assert duck.sent[-1][1] == "stop"


def test_a_duck_that_never_takes_the_posture_ends_the_session_with_a_stop(tmp_path: Path) -> None:
    duck = FakeDuck()
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=False, unit=""))
    stand = next(s for s in R.PLAN if s.name == "floor_stand")
    with pytest.raises(R.SegmentFailed, match="never came"):
        R.record(duck, [stand], sink, Person(duck), drive=False, retries=0)  # (nobody stands it up)
    sink.close()
    assert duck.sent[-1][1] == "stop"


def test_a_robotd_without_its_own_clock_is_refused_at_once(tmp_path: Path) -> None:
    class NoClock(FakeDuck):
        def read_state(self, timeout_s: float) -> dict[str, Any]:
            state = super().read_state(timeout_s)
            del state["t_ns"]
            return state

    duck = NoClock()
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=False, unit=""))
    with pytest.raises(R.SegmentFailed, match="t_ns"):
        R.record(duck, R.PLAN[:1], sink, Person(duck), drive=True, retries=0)
    sink.close()
    assert duck.sent[-1][1] == "stop"


def test_robotds_own_pickup_pause_counts_as_the_claim(tmp_path: Path) -> None:
    duck = FakeDuck()
    duck.picked_up = True  # a real board's [pickup]: in the hand robotd pauses the policy ("held")
    session(tmp_path, duck)
    legs = F.fit(R.read(tmp_path / R.RECORDING)[1])
    assert legs.usable and legs.replay["lifts"]["lift_stand"] == {"claimed": 3, "of": 3}
    assert all(
        row["by_robotd"] for row in legs.replay["rows"] if row["segment"].startswith("lift_")
    )


# ── the ToF touch band ──


def tof_fit(tmp_path: Path, reads: Any) -> T.TofFit:
    duck = FakeDuck()
    duck.tof_reads = reads
    sink = R.Sink(tmp_path / R.RECORDING)
    sink.open(cli.header_for(duck.info, tof=True, unit=""))
    R.record(duck, R.tof_plan(), sink, Person(duck))
    sink.close()
    got = T.fit(R.read(tmp_path / R.RECORDING)[1])
    assert got is not None
    return got


def test_a_sensor_that_reads_like_the_simulators_keeps_its_numbers(tmp_path: Path) -> None:
    got = tof_fit(tmp_path, None)
    assert got.kept_simulator and got.bands == C.SIMULATOR.tof
    rows = {row.mm: row for row in got.rows}
    assert set(rows) == set(R.TOF_DISTANCES_MM)
    assert min(rows[0].counts[100]) == 64 and max(rows[150].counts[100]) == 0


def test_a_sensor_whose_pressed_face_returns_half_its_zones_gets_its_own_count(
    tmp_path: Path,
) -> None:
    def reads(mm: int) -> tuple[list[int], list[int]]:
        distance, status = cast(sensor(0), wall_x=SENSOR_POS[0] + max(mm, 5) / 1000.0)
        if mm <= 30:  # pressed: the outer zones return nothing
            status = [5 if 2 <= i % 8 <= 5 and 1 <= i // 8 <= 6 else 255 for i in range(64)]
        return distance, status

    got = tof_fit(tmp_path, reads)
    assert got.bands is not None and not got.kept_simulator
    assert got.bands.touch_zones == 12 and got.gap == 24  # 24 zones on the board, none clear of it


def test_a_sensor_that_returns_nothing_when_pressed_cannot_use_a_zone_count(tmp_path: Path) -> None:
    def reads(mm: int) -> tuple[list[int], list[int]]:
        distance, status = cast(sensor(0), wall_x=SENSOR_POS[0] + max(mm, 5) / 1000.0)
        return (distance, [255] * 64) if mm <= 30 else (distance, status)

    got = tof_fit(tmp_path, reads)
    assert got.bands is None and "no too-close band" in got.why
    header = cli.header_for(INFO, tof=True, unit="")
    legs = F.Fit(bands=C.SIMULATOR.proprio)
    assert any("ToF touch band" in why for why in cli.why_not(header, legs, got))
    assert cli.build(header, legs, got, "0" * 64) is None
    # a duck without the ToF module needs no ToF measurement; a rehearsal keeps the simulated sensor's band
    assert cli.why_not(cli.header_for(INFO, tof=False, unit=""), legs, None) == []
    rehearsal = {"backend": "real", "rehearsal": True, "policies": INFO["policies"]}
    built = cli.build(cli.header_for(rehearsal, tof=True, unit=""), legs, None, "0" * 64)
    assert built is not None and built.board == C.REHEARSAL_BOARD and built.tof == C.SIMULATOR.tof


# ── over the link ──


@pytest.fixture
def bridged(tmp_path: Path):
    """A fake robotd behind a real bridge that speaks as a rehearsal's does (``backend: real``, its networks)."""
    with tempfile.TemporaryDirectory(prefix="dm-bu-t-", dir=short_tmp()) as directory:
        robotd = StateRobotd(os.path.join(directory, "r.sock"))
        bridge = Bridge(
            listen=("127.0.0.1", 0),
            token=TOKEN,
            channels={"robotd": Channel("robotd", unix=robotd.path)},
            keeper=None,
            body_info={
                "backend": "real",
                "rehearsal": True,
                "policies": {"walk_sha256": WALK, "sitstand_sha256": SEAT},
            },
        )
        bridge.start()
        token = tmp_path / "token"
        token.write_bytes(TOKEN)
        token.chmod(0o600)
        yield SimpleNamespace(robotd=robotd, bridge=bridge, token=token)
        bridge.close()
        robotd.close()


def test_the_tool_reads_the_duck_over_the_bridge_and_says_which_calibration_it_has(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bridged: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from duckmind.bringup.link import LinkBody

    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    url = f"remote://127.0.0.1:{bridged.bridge.port}"
    body = LinkBody(url, bridged.token.read_bytes())
    try:
        assert C.key_of(body.info) == (C.REHEARSAL_BOARD, WALK, SEAT) and not body.has_tof
        states = [body.read_state(2.0) for _ in range(5)]
        assert all(s is not None and s["policy"] == "stand" for s in states)
        body.sit_toggle()
        body.move(0.3)
        body.stop()
    finally:
        body.close()
    sent = [m for m in bridged.robotd.methods if m not in ("hello", "robot.subscribe")]
    # (three connections: the order across them is the fake's, not the tool's)
    assert sorted(sent) == ["robot.do", "robot.move", "robot.stop"]
    assert bridged.robotd.skills == ["sit_toggle"]
    args = ["show", url, "--token-file", str(bridged.token)]
    assert cli.main(args) == 1 and "head only" in capsys.readouterr().out
    C.save(C.parse(_document(C.REHEARSAL_BOARD)))
    assert cli.main(args) == 0 and "calibration:" in capsys.readouterr().out


def test_watch_only_reads_and_writes_what_it_saw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridged: Any
) -> None:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    url = f"remote://127.0.0.1:{bridged.bridge.port}"
    out = tmp_path / "session"
    code = cli.main(
        ["watch", url, "--token-file", str(bridged.token), "--out", str(out), "--minutes", "0.01"]
    )
    (written,) = out.glob("watch-*.json")
    seen = json.loads(written.read_text(encoding="utf-8"))
    assert code == 0 and seen["ticks"] > 10 and seen["policies"] == {"stand": seen["ticks"]}
    assert seen["edges"] == {} and seen["link"]["drops"] == 0
    assert (
        seen["currents_reported"] is False
    )  # (this robotd reports no currents: stage 0's finding)
    assert seen["calibration"] is None and seen["key"][0] == C.REHEARSAL_BOARD
    asked = set(bridged.robotd.methods) - {"hello", "robot.subscribe"}
    assert asked == {"robot.health"}  # nothing that moves the duck


def test_the_record_command_dials_the_bridge_prompts_and_always_ends_with_a_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bridged: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DUCKMIND_HOME", str(tmp_path / "home"))
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "")
    url = f"remote://127.0.0.1:{bridged.bridge.port}"
    out = tmp_path / "session"
    args = ["record", url, "--token-file", str(bridged.token), "--out", str(out), "--stages", "1"]
    # (this fake robotd sends no t_ns: the first segment is refused, three tries, and the session ends)
    assert cli.main(args) == 1
    said = capsys.readouterr()
    assert f"board {C.REHEARSAL_BOARD}" in said.out and "air_sit" in said.out
    assert len(asked) == 3 and "t_ns" in said.err and "told to stop" in said.err
    header, lines = R.read(out / R.RECORDING)
    assert header["key"] == [C.REHEARSAL_BOARD, WALK, SEAT] and lines == []
    sent = [m for m in bridged.robotd.methods if m not in ("hello", "robot.subscribe")]
    assert sorted(set(sent)) == ["robot.move", "robot.stop"] and bridged.robotd.skills == []
    # a session folder recorded on another body is not appended to
    bridged.bridge.body_info["policies"] = {"walk_sha256": "c" * 64, "sitstand_sha256": SEAT}
    assert cli.main(args) == 2 and "another body" in capsys.readouterr().err


class FakeWorld:
    """A world server's hand as the rehearsal's person drives it: ``x_hand`` starts a script that takes three
    ``x_world`` reads to finish (the ops and the hand's fields are world/hand.py's)."""

    def __init__(self) -> None:
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.ops: list[dict[str, Any]] = []
        self.holding, self.pending, self.left = False, None, 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        conn, _ = self.server.accept()
        stream = conn.makefile("rwb")
        for raw in stream:
            ask = json.loads(raw)
            answer: dict[str, Any] = {"ok": True}
            if ask["op"] == "x_hand":
                self.ops.append({k: v for k, v in ask.items() if k != "op"})
                self.pending, self.left = ask["do"], 3
            elif ask["op"] == "x_world":
                if self.pending is not None:
                    self.left -= 1
                    if self.left == 0:
                        self.holding, self.pending = self.pending == "pickup", None
                state = "carry" if self.pending else ("hold" if self.holding else "idle")
                answer = {"hand": {"state": state, "holding": self.holding, "held": self.holding}}
            stream.write((json.dumps(answer) + "\n").encode())
            stream.flush()


def test_a_rehearsals_person_is_the_simulated_worlds_hand() -> None:
    from duckmind.bringup.simperson import SimPerson

    world = FakeWorld()
    person = SimPerson(world.port, say=lambda text: None, pause=lambda s: None)
    by_name = {s.name: s for s in R.PLAN}
    person.ready(by_name["air_stand"], 0)  # held in the air
    assert world.ops == [{"do": "pickup"}] and world.holding
    person.ready(by_name["air_walk"], 0)  # (already in the hand: nothing more)
    assert len(world.ops) == 1
    person.ready(by_name["lift_stand"], 0)  # on the floor, on the family's mat
    assert world.ops[-1] == {"do": "place", "to": "mat"} and not world.holding
    person.cue(by_name["lift_stand"], 0, "lift")
    person.cue(by_name["lift_stand"], 0, "set_down")
    assert world.ops[-2:] == [{"do": "pickup"}, {"do": "place", "to": "here"}]
    person.cue(by_name["cycle"], 0, "sit")  # (the posture is the tool's, not the hand's)
    assert len(world.ops) == 4
    with pytest.raises(R.SegmentFailed, match="no board"):
        person.ready(R.tof_plan()[0], 0)


def _document(board: str) -> dict[str, Any]:
    return C.Calibration(
        board=board,
        walk_sha256=WALK,
        sitstand_sha256=SEAT,
        source="bench",
        proprio=C.SIMULATOR.proprio,
        tof=C.SIMULATOR.tof,
        recorded="2026-11-10T00:00:00Z",
    ).to_json()


def test_the_default_config_is_what_the_fit_replays_with() -> None:
    # (the holds the levels are measured over are proprio's own)
    assert (ProprioConfig().lift_hold_s, ProprioConfig().unsure_hold_s) == (0.30, 0.15)
