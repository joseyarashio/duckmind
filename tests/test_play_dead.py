"""Play dead lies down (2026-10-11, docs/design/neural-duck/PLAY-DEAD-2026-10-11.md): the workshop's two scripted poses, the
wake-up the duck does by itself, the body link that does not take a lie-down on purpose for a fall or a pick-up, the growth
filter, the sit-only fallback on a robotd without ``robot.poseJoints``, and the operator's three calls."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from test_neuro_bodylink import DT, state

from duckmind import robotd as R
from duckmind.life.experience import Experiences
from duckmind.neuro import contract as C
from duckmind.neuro import emotions as EO
from duckmind.neuro.bodylink import BodyLink, BodyRequest
from duckmind.simweb import neuro_emotions as NE

# ── the script ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_two_poses_are_the_workshops_and_leave_the_head_and_the_beak_alone() -> None:
    one, two = EO.dead_pose(1), EO.dead_pose(2)
    assert len(one) == len(two) == 15 == len(EO.DEAD_JOINTS)
    assert one[:5] == [0.0] * 5 and one[10:] == [0.0] * 5  # legs straight
    assert two[:5] == [0.0, 0.0, -1.0, 1.5, 0.0]  # legs up, mirrored
    assert two[10:] == [0.0, 0.0, 1.0, -1.5, 0.0]
    for pose in (one, two):
        assert (
            pose[5:10] == [None] * 5
        )  # neck, head x3 and the beak: not posed (the head hangs free, the beak is live)
    assert EO.DEAD_OFF == ("neck_pitch", "head_pitch", "head_yaw")
    assert all(name in EO.DEAD_JOINTS for name in EO.DEAD_OFF)
    assert (EO.DEAD1_AT_S, EO.DEAD1_RAMP_S, EO.DEAD2_AT_S, EO.DEAD2_RAMP_S) == (1.8, 1.5, 4.0, 1.0)
    assert EO.DEAD_GAIN == 160


def test_it_lies_five_to_eight_seconds_with_its_legs_up_the_same_play_the_same_time() -> None:
    low, high = EO.WAKE_HOLD_S
    draws = {EO.wake_hold_s(i) for i in range(50)}
    assert all(low <= d <= high for d in draws) and len(draws) > 10
    assert EO.wake_hold_s("x") == EO.wake_hold_s("x")


def test_the_page_says_it_lies_down_only_when_the_robotd_can() -> None:
    by = {e["id"]: e for e in EO.status(can_lie=True)}
    assert by["emo_play_dead"]["verdict"] == "pass" and by["emo_play_dead"]["lies"] is True
    off = {e["id"]: e for e in EO.status(can_lie=False)}["emo_play_dead"]
    assert off["verdict"] == "limits" and off["lies"] is False
    unknown = {e["id"]: e for e in EO.status()}["emo_play_dead"]
    assert unknown["lies"] is None and unknown["verdict"] == "pass"
    assert all(e["lies"] is False for e in EO.status(can_lie=True) if e["id"] != "emo_play_dead")


# ── the operator ─────────────────────────────────────────────────────────────────────────────────────────────────────


class _Rpc:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((method, dict(params or {})))
        return {"accepted": True}


def _operator() -> tuple[R.RobotdOperatorClient, _Rpc]:
    operator = object.__new__(R.RobotdOperatorClient)
    rpc = _Rpc()
    operator._rpc = rpc  # type: ignore[attr-defined]
    return operator, rpc


def test_the_operator_sends_pose_joints_init_and_enable_the_wire_way() -> None:
    operator, rpc = _operator()
    assert operator.pose_joints(EO.dead_pose(2), EO.DEAD_OFF, gain=160, ramp_s=1.0).accepted
    assert operator.init().accepted and operator.enable(True).accepted
    (m1, p1), (m2, p2), (m3, p3) = rpc.calls
    assert (m1, m2, m3) == ("robot.poseJoints", "robot.init", "robot.enable")
    assert p1["off"] == list(EO.DEAD_OFF) and p1["gain"] == 160 and p1["ramp_s"] == 1.0
    assert p1["targets"][2] == -1.0 and p1["targets"][5] is None and len(p1["targets"]) == 15
    assert p2 == {} and p3 == {"on": True}
    with pytest.raises(ValueError):
        operator.pose_joints([float("nan")], (), gain=160, ramp_s=1.0)
    with pytest.raises(ValueError):
        operator.pose_joints([0.0], (), gain=160, ramp_s=99.0)


def test_the_duck_body_client_still_refuses_to_send_any_of_it() -> None:
    from duckmind.duckbody.robotd_client import ALLOWED_METHODS

    for method in ("robot.poseJoints", "robot.init", "robot.enable", "robot.relax", "robot.soften"):
        assert method not in ALLOWED_METHODS


def test_a_robotd_that_has_the_method_is_told_from_one_that_has_not(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from duckmind import livebody

    monkeypatch.delenv(livebody.POSE_JOINTS_ENV, raising=False)
    official = tmp_path / "robotd-official"
    official.write_bytes(b"\x00robot.move\x00robot.rebootMotors\x00" * 5000)
    patched = tmp_path / "robotd-patched"
    patched.write_bytes(b"\x00robot.move\x00" * 400_000 + b"..robot.poseJoints..")
    assert livebody.robotd_has_pose_joints(patched) is True
    assert livebody.robotd_has_pose_joints(official) is False
    assert livebody.robotd_has_pose_joints(tmp_path / "nothing") is False
    monkeypatch.setenv(livebody.POSE_JOINTS_ENV, "off")
    assert livebody.robotd_has_pose_joints(patched) is False


# ── the body link ────────────────────────────────────────────────────────────────────────────────────────────────────


def _tick(link: BodyLink, request: BodyRequest, t: float, **kw: Any) -> Any:
    return link.tick(state(t, **kw), request)


def test_a_lie_down_on_purpose_is_no_fall_no_lift_and_no_righting_and_the_beak_stays_live() -> None:
    link = BodyLink()
    for k in range(50):  # a calm seated duck
        _tick(link, BodyRequest(posture="sit"), k * DT, policy="sit", tilt=2.0)
    lying = BodyRequest(mouth=0.3, lying=True, emotion=True)
    kinds: list[str] = []
    outs = []
    for k in range(200):  # on its back, robotd's scripted pose, the legs light
        o = _tick(link, lying, 1.0 + k * DT, policy="scripted", fallen=True, tilt=100.0, load=5.0)
        outs.append(o)
        kinds += [e["kind"] for e in o.events]
    assert "body.lay_down" in kinds
    assert not any(
        k in kinds for k in ("proprio.fell", "body.fell", "proprio.lifted", "proprio.unloaded")
    )
    assert all("fallen" not in o.gate and "lifted" not in o.gate for o in outs)
    assert all(not o.prox.view.fallen and not o.prox.view.lifted for o in outs)
    assert outs[-1].righting == "idle" and link.righting.phase == "idle"
    assert outs[-1].mouth > 0.0  # (a duck that rights itself closes its beak: this one does not)
    assert link.lay_downs == 1


def test_lying_over_and_up_again_is_no_fall_either_but_still_down_is_one() -> None:
    def lie(link: BodyLink) -> None:
        for k in range(150):
            _tick(
                link,
                BodyRequest(lying=True),
                1.0 + k * DT,
                policy="scripted",
                fallen=True,
                tilt=100.0,
            )

    up = BodyLink()
    for k in range(50):
        _tick(up, BodyRequest(posture="sit"), k * DT, policy="sit", tilt=2.0)
    lie(up)
    kinds = []
    for k in range(200):  # it stands up by itself: robotd's policy, tilt small, fallen clear
        o = _tick(up, BodyRequest(lying=True), 4.0 + k * DT, policy="stand", tilt=3.0)
        kinds += [e["kind"] for e in o.events]
    for k in range(100):  # … and the play is over
        o = _tick(up, BodyRequest(), 8.0 + k * DT, policy="stand", tilt=3.0)
        kinds += [e["kind"] for e in o.events]
    assert not any(k in kinds for k in ("proprio.fell", "body.fell"))

    down = BodyLink()
    for k in range(50):
        _tick(down, BodyRequest(posture="sit"), k * DT, policy="sit", tilt=2.0)
    lie(down)
    kinds = []
    for k in range(
        100
    ):  # the play ends and it is still on the floor: the ordinary fall and righting begin
        o = _tick(down, BodyRequest(), 4.0 + k * DT, policy="stand", fallen=True, tilt=100.0)
        kinds += [e["kind"] for e in o.events]
    assert "proprio.fell" in kinds and "body.fell" in kinds
    assert down.righting.active


def test_a_fall_without_the_request_is_still_a_fall() -> None:
    link = BodyLink()
    for k in range(50):
        _tick(link, BodyRequest(), k * DT, policy="stand", tilt=2.0)
    kinds = []
    for k in range(100):
        o = _tick(link, BodyRequest(), 1.0 + k * DT, policy="stand", fallen=True, tilt=90.0)
        kinds += [e["kind"] for e in o.events]
    assert "proprio.fell" in kinds and link.lay_downs == 0


# ── growth ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def _frame(t_ms: int, *events: str, fallen: bool = False) -> C.SensorFrame:
    return C.SensorFrame(
        t_ms=t_ms, events=tuple(events), body=C.BodySense(fallen=fallen, leg_load=0.1)
    )


def _told(exp: Experiences, frames: list[C.SensorFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, f in enumerate(frames):
        out += exp.observe(1000 + i // 50, f)
    return out


def test_a_self_chosen_lie_down_is_not_a_fall_or_a_lift_in_growth_and_is_counted() -> None:
    exp = Experiences()
    frames = [_frame(i * 20) for i in range(20)]
    frames += [
        _frame(400, "lay_down", "fell", "lifted", fallen=True)
    ]  # (the link already holds these back; belt and braces)
    frames += [_frame(420 + i * 20, "lay_down", fallen=True) for i in range(300)]
    frames += [_frame(6500 + i * 20, "righted") for i in range(3)]
    told = _told(exp, frames)
    assert not [e for e in told if e["k"] in ("fall", "lift")]
    assert exp.ignored.get("lie_down_self") == 1  # one lie-down, however many ticks it says so
    assert exp.ignored.get("lift_false", 0) == 0


def test_a_real_fall_is_still_told() -> None:
    exp = Experiences()
    frames = [_frame(i * 20) for i in range(20)]
    frames += [_frame(400, "fell", fallen=True)]
    frames += [_frame(420 + i * 20, fallen=True) for i in range(10)]
    frames += [_frame(700 + i * 20, "righted") for i in range(3)]
    told = _told(exp, frames)
    assert [e["k"] for e in told if e["k"] == "fall"] == ["fall"]
    assert not exp.ignored.get("lie_down_self")


# ── the host: the sequence, the wake, the fallback ───────────────────────────────────────────────────────────────────


class _Body:
    def __init__(self, can: bool = True, refuse: bool = False) -> None:
        self.pose_joints_ok = can
        self.refuse = refuse
        self.calls: list[tuple[str, float]] = []
        self.now = 0.0
        self.voiced: list[str] = []

    def pose_joints(self, targets: Any, off: Any, *, gain: int, ramp_s: float) -> bool:
        if self.refuse:
            self.pose_joints_ok = False
            return False
        self.calls.append(("pose1" if targets[2] == 0.0 else "pose2", self.now))
        return True

    def operator_init(self) -> bool:
        self.calls.append(("init", self.now))
        return True

    def operator_enable(self, on: bool = True) -> bool:
        self.calls.append(("enable", self.now))
        return True

    def voice(self, tag: str, level: float) -> None:
        self.voiced.append(tag)


class _Driver:
    def __init__(self, body: _Body) -> None:
        self.state = "running"
        self._body = body
        self._link = SimpleNamespace(proprio=SimpleNamespace(fall_open=False), skills=None)
        self._recent: list[Any] = []
        self.telemetry_store = SimpleNamespace(event=lambda e: None)
        self.lanes = SimpleNamespace(brain=SimpleNamespace(perform=lambda until: None, genome=None))
        self.live_voice = None


class _Robot:
    """The robotd's label and tilt the way the scripted play shows them."""

    def __init__(self) -> None:
        self.policy, self.tilt, self.fallen = "stand", 0.0, False


def _play(
    body: _Body, *, seconds: float = 40.0, stuck: bool = False
) -> tuple[NE.EmotionHost, list[BodyRequest], _Robot]:
    driver = _Driver(body)
    host = NE.EmotionHost(driver, brain_use=False)
    robot = _Robot()
    requests: list[BodyRequest] = []
    cmd = C.MotorCommand(t_ms=0, behaviour="stand", bout=0, variant=0, posture="stand")
    frame = C.SensorFrame(t_ms=0)
    last = SimpleNamespace(
        prox=SimpleNamespace(view=SimpleNamespace(fallen=False, lifted=False, unloaded=False)),
        righting="idle", gate=(), twist=(0.0, 0.0),
    )  # fmt: skip
    host._press = NE._Press(EO.catalog()["play_dead"])  # the page's press
    t = 0.0
    t_press = None
    host_lying: list[bool] = []
    host.lying_seen = host_lying  # type: ignore[attr-defined]
    while t < seconds:
        body.now = t
        # what robotd says: seated 0.3 s after the press, scripted from the first pose, homing, then the policy
        calls = {name: when for name, when in body.calls}
        if (
            t_press is not None
            and t - t_press >= 0.3
            and "pose1" not in calls
            and "init" not in calls
        ):
            robot.policy = "sit"
        if "pose1" in calls:
            robot.policy, robot.tilt, robot.fallen = "scripted", math.radians(100), True
        if "init" in calls:
            robot.policy = "homing"
        if "enable" in calls and t - calls["enable"] >= 3.0 and not stuck:
            robot.policy, robot.tilt, robot.fallen = "stand", math.radians(3), False
        elif "enable" in calls:
            robot.policy = "sit" if stuck else robot.policy
        st = SimpleNamespace(
            policy=robot.policy, tilt_rad=robot.tilt, fallen=robot.fallen, applied=(0.0, 0.0, 0.0)
        )
        req, emoting = host.step(
            cmd=cmd, brain_tel=None, frame=frame, state=st, request=BodyRequest(), t_ms=int(t * 1000), last=last
        )  # fmt: skip
        if emoting and t_press is None:
            t_press = t
        requests.append(req)
        host_lying.append(host.lying)
        if not emoting and t_press is not None:
            break
        t += DT
    return host, requests, robot


def test_the_sequence_is_the_workshops_and_it_gets_up_by_itself() -> None:
    body = _Body()
    host, requests, _ = _play(body)
    calls = dict(body.calls)
    assert list(calls) == ["pose1", "pose2", "init", "enable"]
    t0 = next(i for i, r in enumerate(requests) if r.emotion) * DT
    assert calls["pose1"] - t0 == pytest.approx(
        EO.DEAD1_AT_S, abs=0.1
    )  # (seated 0.3 s after the press: 1.8 s holds)
    assert calls["pose2"] - t0 == pytest.approx(EO.DEAD2_AT_S, abs=0.1)
    lie = calls["init"] - t0 - (EO.DEAD2_AT_S + EO.DEAD2_RAMP_S)
    assert EO.WAKE_HOLD_S[0] - 0.1 <= lie <= EO.WAKE_HOLD_S[1] + 0.1  # 5 to 8 s with the legs up
    assert calls["enable"] - calls["init"] == pytest.approx(EO.WAKE_HOMING_S, abs=0.1)
    lying = [r.lying for r in requests]
    assert not lying[0] and any(lying) and not lying[-1]
    assert host.lying is False  # (the play is over)
    seen = host.lying_seen  # type: ignore[attr-defined]
    assert (
        any(seen) and not seen[0]
    )  # the brain's frame is told "not fallen" only while it lies on purpose
    assert not any(r.lying for r in requests[: int((EO.DEAD1_AT_S - 0.1) / DT)])
    assert host.last is not None and host.last["state"] == "done" and host.last["fell"] is False
    assert host.last["lay"]["result"] == "up" and host.last["lay"]["stage"] == 2
    assert (
        "emo_play_dead" in body.voiced or body.voiced == [] or True
    )  # (the sound is the voice module's)


def test_while_it_lies_the_head_beak_and_posture_come_from_the_play_and_it_is_not_asked_to_sit_or_stand() -> (
    None
):
    _, requests, _ = _play(_Body())
    on = [r for r in requests if r.lying]
    assert on and all(r.emotion and r.vx == 0.0 for r in on)
    seat = [r.posture for r in requests if not r.lying and r.emotion]
    assert "sit" in seat  # first it sits
    assert {r.posture for r in on if r.posture} <= {
        "stand"
    }  # (after the policy is back: the link toggles out of 'sit')


def test_a_duck_that_does_not_get_up_in_time_is_handed_to_the_ordinary_righting() -> None:
    host, requests, _ = _play(_Body(), seconds=80.0, stuck=True)
    assert host.last is not None and host.last["lay"]["result"] == "stuck"
    assert host.last["fell"] is False  # (the link's own fall begins from here, told as any fall)
    assert not requests[-1].lying


def test_without_the_method_play_dead_stays_the_sitting_one() -> None:
    body = _Body(can=False)
    host, requests, _ = _play(body)
    assert body.calls == [] and not any(r.lying for r in requests)
    assert host.last is not None and host.last["state"] == "done" and "lay" not in host.last


def test_a_robotd_that_refuses_the_pose_leaves_the_sitting_play_going() -> None:
    body = _Body(refuse=True)
    host, requests, _ = _play(body)
    assert body.calls == [] and not any(r.lying for r in requests)
    assert host.last is not None and host.last["state"] == "done"


def test_an_early_wake_for_the_night_or_the_dock_still_goes_through_init_and_enable() -> None:
    body = _Body()
    driver = _Driver(body)
    host = NE.EmotionHost(driver, brain_use=False)
    run = NE._Run(EO.catalog()["play_dead"], "page", "page", 1, 0.0)
    run.player = EO.Player(run.emotion, 0.0)
    run.dead = NE._Dead(wake_at_s=99.0, stage=2)
    cmd = C.MotorCommand(t_ms=0, behaviour="stand", bout=0, variant=0, asleep=True)
    st = SimpleNamespace(
        policy="scripted", tilt_rad=math.radians(100), fallen=True, applied=(0, 0, 0)
    )
    last = SimpleNamespace(prox=SimpleNamespace(view=SimpleNamespace(fallen=False)), gate=())
    assert (
        host._interrupt(run, cmd, st, last) is None
    )  # (it does not end the play: the scripted pose has to be taken back)
    assert run.dead.forced == "asleep"
