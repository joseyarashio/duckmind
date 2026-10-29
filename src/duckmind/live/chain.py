"""The whole chain on a fake clock (live voice, LV2): a recording, the gate, the bridge, a brain.

``run_chain`` plays one of the LV1 cassettes the way the duck would live it: the scenario's
microphone frames feed the session (and the duck's local voice detection), the session's replayed
tool calls are judged by the grounding gate, the bridge makes candidates of what got through and
a real integrated brain, stepping at 25 Hz on the scenario's clock, decides. Nothing here touches
the network or the physics; the world is the stand-in of the scenarios (the duck sits at the
origin looking along +x, the family stand where the scenario says), and **what the duck perceives**
is a small scripted scene (:data:`SCENES`: who is in its camera's view when, with a look of their
own, and what thing) taken from the scenario's own geometry — a person is in view when the
camera, 37.4° wide, would see them. That script is the honest limit of the replay: the duck's eye
and its person re-identification are LV3's, so who it perceives is stated here, not rendered.

Two more things for the tests and the notes: ``inject`` adds tool calls to the recording at given
moments (a misheard name, a call outside the bank: what the real model did not happen to say), and
:func:`closed_loop_barge` runs a call and a person speaking over it with the duck's own echo in the
microphone and the stop cutting that echo, to measure the stop in time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.live import events as E
from duckmind.live import scenario as S
from duckmind.live.bridge import BridgeConfig, LiveBridge
from duckmind.live.cassette import Cassette, ReplayTransport
from duckmind.live.echo import CallMark
from duckmind.live.meter import UsageMeter
from duckmind.live.session import LiveSession, QueuedCall
from duckmind.live.tools import parse_call
from duckmind.live.vad import BankReferences, VoiceDetector
from duckmind.neuro import contract as C
from duckmind.neuro.genome import Genome

FOV_HALF_RAD = math.radians(37.4) / 2.0  # the camera's half width
BRAIN_HZ = 25.0
FRAME_S = 0.02

# what the eye gives of each person (legs r, g, luma, length, shirt r, g): well apart
LOOKS: dict[str, tuple[float, ...]] = {
    "mom": (0.30, 0.28, 0.35, 0.80, 0.55, 0.30),
    "dad": (0.20, 0.30, 0.20, 0.90, 0.25, 0.30),
    "child": (0.50, 0.30, 0.55, 0.40, 0.30, 0.45),
}
ALL = 1e9


@dataclass(frozen=True)
class PersonPlan:
    who: str  # a key of LOOKS
    track: int
    at_m: tuple[float, float]  # from the duck (x ahead, y to its left)
    from_s: float = 0.0
    until_s: float = ALL


@dataclass(frozen=True)
class ThingPlan:
    track: int
    kind: str  # "toy" | "thing"
    at_m: tuple[float, float]
    from_s: float = 0.0
    until_s: float = ALL


@dataclass(frozen=True)
class ScenePlan:
    people: tuple[PersonPlan, ...] = ()
    things: tuple[ThingPlan, ...] = ()


# who is in front of the duck in each scenario (the places are the scenario's: a person who speaks
# from outside the camera's view is not in it)
SCENES: dict[str, ScenePlan] = {
    "silence": ScenePlan(),
    "call_name": ScenePlan((PersonPlan("mom", 1, (2.0, 0.5)),)),
    "hello": ScenePlan((PersonPlan("child", 2, (1.5, -0.5)),)),
    "other_name": ScenePlan((PersonPlan("mom", 1, (2.0, 0.5)),)),
    "stop_calling": ScenePlan((PersonPlan("mom", 1, (1.5, 0.0)),)),
    "echo_only": ScenePlan(),
    # dad says "this is mom" from beside the duck (outside its view) while mom stands in front
    "intro_then_hello_mom": ScenePlan((PersonPlan("mom", 1, (2.0, 0.0)),)),
    # … and in the other one the child comes into view after mom has gone
    "intro_then_hello_kid": ScenePlan(
        (
            PersonPlan("mom", 1, (2.0, 0.0), 0.0, 9.0),
            PersonPlan("child", 2, (1.5, -0.5), 9.5),
        )
    ),
    "what_do_you_see": ScenePlan(
        (PersonPlan("dad", 3, (1.5, 0.0)),), (ThingPlan(7, "toy", (0.55, 0.0)),)
    ),
    # (the home's far toys at the wall are in the picture but the duck's eye does not report them)
    "what_do_you_see_empty": ScenePlan((PersonPlan("dad", 3, (1.5, 0.0)),)),
}


def scene_for(scenario_id: str) -> ScenePlan:
    return SCENES.get(scenario_id, ScenePlan())


def _polar(at_m: tuple[float, float]) -> tuple[float, float]:
    return math.atan2(at_m[1], at_m[0]), math.hypot(at_m[0], at_m[1])


def sightings_at(plan: ScenePlan, t: float) -> list[C.Sighting]:
    """What the camera reports at ``t``: the people and things inside its field of view."""
    out: list[C.Sighting] = []
    for p in plan.people:
        if p.from_s <= t < p.until_s:
            bearing, rng = _polar(p.at_m)
            if abs(bearing) <= FOV_HALF_RAD:
                out.append(
                    C.Sighting(
                        kind="person",
                        bearing=bearing,
                        range_m=rng,
                        size=0.3,
                        track=p.track,
                        source="model",
                        look=LOOKS[p.who],
                    )
                )
    for th in plan.things:
        if th.from_s <= t < th.until_s:
            bearing, rng = _polar(th.at_m)
            if abs(bearing) <= FOV_HALF_RAD:
                out.append(
                    C.Sighting(
                        kind=th.kind,
                        bearing=bearing,
                        range_m=rng,
                        size=0.08,
                        track=th.track,
                        source="eye",
                    )
                )
    return out


@dataclass(frozen=True)
class Injected:
    """A tool call added to a recording at a moment of its clock."""

    t: float
    name: str
    args: dict[str, Any]
    id: str = ""


def with_injected(cassette: Cassette, extra: Sequence[Injected]) -> Cassette:
    """A copy of the recording with more tool calls in it (the key and the rest are the same)."""
    events = list(cassette.events)
    for k, item in enumerate(extra):
        call = {"id": item.id or f"injected-{k}", "name": item.name, "args": dict(item.args)}
        events.append(E.LiveEvent(E.TOOL_CALL, item.t, {"calls": [call]}))
    return replace(cassette, events=sorted(events, key=lambda e: e.t))


class _CapturingRun(S.ScenarioRun):
    """A scenario run that keeps the microphone it makes (its vocal windows are the duck's own)."""

    mic: Any = None

    def make_mic(self) -> Any:
        self.mic = super().make_mic()
        return self.mic


@dataclass
class ChainResult:
    run: S.ScenarioRun
    brain: Any
    bridge: LiveBridge
    session: LiveSession
    cmds: list[Any]
    plan: ScenePlan
    injected: tuple[Injected, ...] = ()
    mic: Any = None
    ticks: int = 0
    taps: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def summary(self) -> dict[str, Any]:
        gate = self.bridge.gate
        return {
            "run": self.run_id,
            "calls": dict(sorted(gate.counts.items())),
            "gate_refusals": dict(sorted(gate.refusals.items())),
            "entered": len(self.bridge.entered),
            "voiced": len(self.bridge.voiced),
            "intents": {
                k: v for k, v in sorted(self.bridge.counts.items()) if k.startswith("intent")
            },
            "pending": [e for e in self.bridge.log if e["decision"] == "pending"],
            "people": self.brain.learn.people_view(),
            "barges": [b.view() for b in self.bridge.barges],
            "faults": self.bridge.faults,
        }


def brain_for(run: S.ScenarioRun) -> Any:
    """The scenario's duck: the genome of its temperament, its starting drives."""
    from duckmind.neuro.integration import make_brain

    t = run.temperament
    brain = make_brain(Genome.from_seed(t.genome_seed, gd=t.gd), seed=run.seed)
    for name, value in run.drives_for(run.context_at_start()).items():
        if hasattr(brain.drives, name):
            setattr(brain.drives, name, float(value))
    return brain


def run_chain(
    file: S.ScenarioFile,
    scenario_id: str,
    temperament_id: str,
    seed: int,
    *,
    cassette: Cassette | None = None,
    inject: Sequence[Injected] = (),
    config: BridgeConfig | None = None,
    use_vad: bool = True,
    plan: ScenePlan | None = None,
    cloud_off_at_first_call: bool = False,
) -> ChainResult:
    """Play one recorded scenario through session → gate → bridge → brain (see the module doc)."""
    run = _CapturingRun(file, scenario_id, temperament_id, seed, require_bank=False)
    if cassette is None:
        cassette = Cassette.load(S.cassette_path(run.run_id))
    if inject:
        cassette = with_injected(cassette, inject)
    clock = S.ManualClock()
    transport = ReplayTransport(cassette, run.inputs())
    session = LiveSession(
        transport, run.soul_text(), clock=clock, config=run.session_config, meter=UsageMeter(None)
    )
    brain = brain_for(run)
    taps: list[tuple[float, dict[str, float]]] = []  # what the selector was ever given from outside
    original = brain.sel.set_extra
    bridge: LiveBridge | None = None

    def tap(inputs: dict[str, float], **kw: Any) -> dict[str, float]:
        # (what the cloud's accepted calls made the selector want: the duck's own wishes of LV4, which
        # the bridge asks for by itself, are told apart by the bridge and are not the cloud's)
        own = bridge.last_wishes if bridge is not None else {}
        cloud = {k: v - own.get(k, 0.0) for k, v in inputs.items() if v - own.get(k, 0.0) > 1e-9}
        if cloud:
            taps.append((clock.t, cloud))
        return original(inputs, **kw)  # type: ignore[no-any-return]

    brain.sel.set_extra = tap
    detector = VoiceDetector(BankReferences(run.bank(), run.voice_seed())) if use_vad else None
    bridge = LiveBridge(
        session,
        config=config or BridgeConfig(duck_name=file.duck_name),
        vad=detector,
        clock=clock,
    ).attach(brain)
    plan = plan if plan is not None else scene_for(scenario_id)
    sounds = _sounds(run)
    cmds: list[Any] = []
    every = max(1, int(round(1.0 / (BRAIN_HZ * FRAME_S))))
    ticks = 0
    for i, step in enumerate(run.steps()):
        clock.set(step.t)
        if step.context is not None:
            session.set_context(run.snapshot(step.context))
        session.feed_audio(step.frame)
        for _fixture, jpeg in step.images:
            session.feed_image(jpeg)
        session.pump()
        if cloud_off_at_first_call and session.enabled and session.peek_calls():
            session.set_enabled(False)  # (the switch is thrown the moment the model's calls arrive)
        bridge.hear(step.frame, run.mic.vocal_windows() if run.mic is not None else ())
        if i % every == 0:
            t_ms = int(round(step.t * 1000))
            cmd, _tel = brain.step(t_ms, _sensor_frame(plan, sounds, step.t, t_ms))
            cmds.append(cmd)
            ticks += 1
    session.close()
    failure = getattr(transport, "failure", None)
    if failure is not None:
        raise failure
    return ChainResult(run, brain, bridge, session, cmds, plan, tuple(inject), run.mic, ticks, taps)


def _sounds(run: S.ScenarioRun) -> list[tuple[float, float, str, float]]:
    """(from, until, kind, level dB at the duck) of every line the family says (the hearing
    model's input: the world's own level function)."""
    out: list[tuple[float, float, str, float]] = []
    for say in run.scenario.say:
        geo = run.say_geometry(say)
        out.append((say.t, say.t + float(geo["duration_s"]), str(geo["kind"]), float(geo["db"])))
    return out


def _sensor_frame(
    plan: ScenePlan, sounds: Sequence[tuple[float, float, str, float]], t: float, t_ms: int
) -> C.SensorFrame:
    kind: str | None = None
    level = 30.0
    onset = 0.0
    for t0, t1, k, db in sounds:
        if t0 <= t < t1:
            kind, level = k, db
            onset = 1.0 if t - t0 < 0.05 else 0.0
    return C.SensorFrame(
        t_ms=t_ms,
        body=C.BodySense(posture="sit"),
        camera=C.CameraSense(usable=True, sightings=tuple(sightings_at(plan, t))),
        sound=C.SoundSense(level_db=level, kind=kind, onset=onset),
        power=C.PowerSense(battery=0.8),
        day_minute=600.0,
    )


# ── a call and a person speaking over it, closed loop ─────────────────────────────────

# how long a bank call lasts when the microphone has no wav for it (neuro_mic.CALL_S)
CALL_S: dict[str, float] = {"alarm": 0.5, "greet": 0.6, "inquire": 0.5, "chirp": 0.3, "coo": 0.8}
SPEECH_DB = 66.0  # the scripted 不要叫了 from 1.5 m (the world's own level function)


@dataclass(frozen=True)
class BargeTrial:
    """One closed-loop trial: a call, a person over it, the stop."""

    temperament: str
    speak_at: float | None  # when the person began (the microphone's clock); None: nobody did
    call_at: float | None  # when the duck's call began
    stop_at: float | None  # when the stop reached the body
    echo_end: float | None  # when the duck's own voice in the microphone ended
    call_end_natural: float | None  # when the call would have ended without the stop
    detect_s: float | None  # the detector's debounce (its onset → its decision)
    latency_s: float | None  # the person's first sound → the stop reached the body
    stopped: bool

    def view(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _vocalize_call(seq: int, tag: str, level: float, t: float) -> QueuedCall:
    parsed = parse_call("vocalize", {"tags": [{"tag": tag, "level": level, "at_ms": 0}]})
    return QueuedCall(
        seq,
        f"c{seq}",
        parsed.name,
        parsed.args,
        parsed.raw,
        parsed.problems,
        parsed.off_bank,
        t,
        0.0,
        1,
        CallMark(False, 1.0, 0.0, False, False),
        False,
    )


def closed_loop_barge(
    file: S.ScenarioFile,
    temperament_id: str,
    *,
    speak_offset_s: float | None,
    tag: str = "coo",
    level: float = 0.9,
    seed: int = 7,
    quiet_s: float = 12.0,
    run_s: float = 6.0,
    reply_weight: float = 12.0,
) -> tuple[BargeTrial, LiveBridge, Any]:
    """The duck voices a call the cloud proposed; its own echo is in the microphone; a person
    speaks (the scripted 不要叫了, 1.5 m away) ``speak_offset_s`` after the call began (None:
    nobody does); the local detection and the bridge stop the call; the stop cuts the echo.
    The proposal is given a large ``reply_weight`` so that every temperament voices it: this is
    about the stop, not about whether the duck answers. Returns the measures, the bridge and the
    microphone."""
    run = S.ScenarioRun(file, "echo_only", temperament_id, seed, require_bank=True)
    mic = run.make_mic()
    detector = VoiceDetector(BankReferences(run.bank(), run.voice_seed()))
    clock = S.ManualClock()
    brain = brain_for(run)
    stops: list[float] = []

    def cut(_request: Any) -> None:
        stops.append(clock.t)
        mic.cut_calls()

    bridge = LiveBridge(
        None,
        config=BridgeConfig(duck_name=file.duck_name, reply_weight=reply_weight),
        vad=detector,
        clock=clock,
        on_stop=cut,
    ).attach(brain)
    every = max(1, int(round(1.0 / (BRAIN_HZ * FRAME_S))))
    t = 0.0
    calls: list[dict[str, Any]] = []
    call_at: float | None = None
    speak_at: float | None = None
    submitted = False
    n = int(round((quiet_s + run_s) / FRAME_S))
    for i in range(1, n + 1):
        t = round(i * FRAME_S, 6)
        clock.set(t)
        if not submitted and t >= quiet_s:
            bridge.submit(_vocalize_call(1, tag, level, t))
            submitted = True
        sounds: list[dict[str, Any]] = []
        if speak_at is not None:
            sounds = [
                {
                    "id": 1,
                    "kind": "voice",
                    "t0": speak_at,
                    "t1": speak_at + 0.72,
                    "db": SPEECH_DB,
                    "src": "person:mom",
                    "utterance": "stop_calling",
                }
            ]
        frame = mic.tick(t, sounds, calls=calls, wall=t)
        calls = []
        bridge.hear(frame, mic.vocal_windows())
        if i % every == 0:
            t_ms = int(round(t * 1000))
            cmd, _ = brain.step(t_ms, _sensor_frame(ScenePlan(), (), t, t_ms))
            if cmd.vocal:
                calls.append({"tag": cmd.vocal, "level": cmd.vocal_level})
                if (
                    call_at is None and len(bridge.voiced) > 0
                ):  # (the cloud's call, not its own chirp)
                    call_at = t
                    if speak_offset_s is not None:
                        speak_at = round(t + speak_offset_s, 2)
    ours = [w for w in mic.vocal_windows() if call_at is not None and w.t0 >= call_at]
    window = ours[0] if ours else None
    barge = bridge.barges[0] if bridge.barges else None
    natural = None
    if window is not None and window.t1 is not None:
        natural = window.t0 + float(CALL_S.get(window.tag, 0.5)) if barge is not None else window.t1
    trial = BargeTrial(
        temperament=temperament_id,
        speak_at=speak_at,
        call_at=call_at,
        stop_at=stops[0] if stops else None,
        echo_end=None if window is None else window.t1,
        call_end_natural=natural,
        detect_s=None if barge is None else barge.detect_s,
        latency_s=None if not stops or speak_at is None else stops[0] - speak_at,
        stopped=bool(barge and barge.stopped),
    )
    return trial, bridge, mic


__all__ = [
    "LOOKS",
    "SCENES",
    "BargeTrial",
    "ChainResult",
    "Injected",
    "PersonPlan",
    "ScenePlan",
    "ThingPlan",
    "brain_for",
    "closed_loop_barge",
    "run_chain",
    "scene_for",
    "sightings_at",
    "with_injected",
]
