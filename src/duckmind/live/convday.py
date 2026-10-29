"""The scripted conversation day (live voice, LV4): the new drives measured in SynthWorld.

A real :class:`~duckmind.neuro.integration.IntegratedBrain` with the live bridge attached lives a
590 s day in :class:`~duckmind.neuro.synth.SynthWorld` (a room, three toys, no family of its own) while
a **script** brings the family: a guest it has never met, mom and dad (introduced by a line from out of
view, so the learner gets a named record), who stand in front of the duck, say addressed lines (the duck's
name, a hello), pet it, and one of them makes a bang. The fake Gemini (:mod:`duckmind.live.fake`)
proposes the replies, 0.9 s after each line as it does in its own delays, and the proposals go through the
real grounding gate and the bridge like any call from the cloud. Everything the duck knows of the day
reaches it through its own senses: a sound kind and a level, a person's sighting with a look, a touch.
Nothing here reads the duck's inside to decide what the family does.

What is measured, the targets and their definitions are frozen in ``docs/design/live-voice/
LV4-TARGETS.md`` (and ``targets_report`` below is those words as code). The honest limits: the audio path
(the echo marks, the voice detection) is LV1/LV2's and is not in this loop (the calls arrive with a clean
mark); the persons are where the script puts them (in front of the duck's camera axis when they speak,
walking there in the second before), not a simulated household; the fake proposes a reply to every line
that is addressed to the duck and a model would not always.

``run_day`` returns plain data (``DayRun.to_dict``) so a pool of processes can run many; the tools
(``tools/live_voice/lv4_search.py``) do the searching and the reporting on it.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from duckmind.live.bridge import BridgeConfig, LiveBridge
from duckmind.live.drives import FEAR_MUTE, DriveParams
from duckmind.live.echo import CallMark
from duckmind.live.fake import REPLY_DELAY_S, TEXT_DELAY_S, FakeGemini
from duckmind.live.session import QueuedCall
from duckmind.live.tools import parse_call
from duckmind.neuro import contract as C
from duckmind.neuro.skill_imprint import LOOK_SCALE
from duckmind.neuro.synth import SoundEvent, SynthWorld

NAME = "蛋黃"
HZ = 10.0
DAY_S = 590.0
NIGHT_S = 150.0
TRAIN_SEEDS = tuple(range(1, 13))
HELD_OUT_SEEDS = tuple(range(13, 29))
PRESETS = ("bold_explorer", "shy_snuggler", "lazy_dreamer", "social_songbird")
TAG_COUNTS = (0.5, 0.3, 0.2)  # P(1, 2, 3 tags) of a proposal: a stress, see the targets file
JITTER_S = 1.5
VIEW_RANGE_M = 1.4  # where a person stands in front of the duck's camera axis
WALK_M_S = 1.5  # how fast they step in front of the duck
UNSEEN_RESET_S = 3.0  # a person the duck has not seen for this long steps in front of it again
RE_APPROACH_M = 2.3  # farther than this from the duck they come back to it
FRIGHT_REFLEXES = frozenset({"startle", "withdraw", "freeze"})
NIGHT_DAY_MINUTE = 1380.0
DAY_MINUTE = 600.0

# what the eye gives of each person (legs r, g, luma, length, shirt r, g): well apart
LOOKS: dict[str, tuple[float, ...]] = {
    "mom": (0.30, 0.28, 0.35, 0.80, 0.55, 0.30),
    "dad": (0.20, 0.30, 0.20, 0.90, 0.25, 0.30),
    "guest": (0.50, 0.30, 0.55, 0.40, 0.30, 0.45),
}
LOOK_NOISE = 0.4  # × the look's own scale: a little under the eye's scatter in good light

WARM_LINES = ("蛋黃，早安！", "蛋黃乖～", "蛋黃，你好嗎？", "蛋黃，今天好嗎？", "蛋黃，過來玩！")
HELLO_LINES = ("你好！", "嗨，你好嗎？", "哈囉～", "你好呀！")
CHATTER = ("晚餐要吃什麼？", "我等一下去買東西。", "今天天氣不錯。")
APOLOGY = "蛋黃，對不起啦。"


# ── the script ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Line:
    """One thing said (``t``: when it starts, on the day's clock)."""

    t: float
    text: str
    kind: str  # the sound kind of its start: "name" | "hello" | "voice"
    who: str  # "guest" | "mom" | "dad" | "" (said from out of view)
    group: str  # guest | mom | dad_pre | dad_bang | dad_post | intro | chat
    ordinal: int = 0  # among the addressed lines of that person (1-based; 0: not addressed)
    addressed: bool = False
    rigid: bool = False  # kept to its second (not jittered)

    @property
    def dur(self) -> float:
        return 0.9 + 0.12 * len(self.text)

    @property
    def end(self) -> float:
        return self.t + self.dur


@dataclass(frozen=True)
class Visit:
    who: str
    t_in: float
    t_out: float


@dataclass(frozen=True)
class Script:
    kind: str
    length_s: float
    day_minute0: float
    dark: bool
    visits: tuple[Visit, ...]
    lines: tuple[Line, ...]
    touches: tuple[tuple[float, float], ...] = ()
    bang_t: float | None = None


def make_script(seed: int, kind: str = "day") -> Script:
    """The day (or the night run) of ``seed``: its times jittered, the order kept."""
    rng = random.Random(f"convday:{kind}:{seed}")

    def jit(t: float, rigid: bool = False) -> float:
        return t if rigid else round(t + rng.uniform(-JITTER_S, JITTER_S), 2)

    lines: list[Line] = []

    def say(t: float, who: str, group: str, text: str, kind_: str, **kw: Any) -> None:
        lines.append(Line(jit(t, kw.get("rigid", False)), text, kind_, who, group, **kw))

    def warm(who: str, group: str, t: float, ordinal: int, rigid: bool = False) -> None:
        text = APOLOGY if rigid else WARM_LINES[(ordinal + seed) % len(WARM_LINES)]
        say(t, who, group, text, "name", ordinal=ordinal, addressed=True, rigid=rigid)

    def chat(t: float, who: str) -> None:
        say(t, who, "chat", CHATTER[int(t) % len(CHATTER)], "voice")

    if kind == "night":
        for k in range(6):
            warm("mom", "mom", 70.0 + 15.0 * k, k + 1)
        return Script(
            "night", NIGHT_S, NIGHT_DAY_MINUTE, True, (Visit("mom", 60.0, NIGHT_S),), tuple(lines)
        )

    # guest (a stranger): quiet presence, then four hello lines
    for k, t in enumerate((52.0, 64.0, 76.0, 88.0)):
        say(
            t,
            "guest",
            "guest",
            HELLO_LINES[(k + seed) % len(HELLO_LINES)],
            "hello",
            ordinal=k + 1,
            addressed=True,
        )
    chat(70.0, "guest")
    # mom, first visit: introduced from out of view, twelve warm lines, two pettings
    say(128.0, "", "intro", "這是媽媽。", "voice")
    for k in range(12):
        warm("mom", "mom", 162.0 + 12.0 * k, k + 1)
    chat(226.0, "mom")
    chat(268.0, "mom")
    touches = (jit(200.0), jit(250.0))
    # dad: introduced, three lines, a bang and an apology, three lines
    say(333.0, "", "intro", "這是爸爸。", "voice")
    for k, t in enumerate((362.0, 374.0, 386.0)):
        warm("dad", "dad_pre", t, k + 1)
    warm("dad", "dad_bang", 399.0, 4, rigid=True)
    for k, t in enumerate((414.0, 426.0, 438.0)):
        warm("dad", "dad_post", t, 5 + k)
    # mom again: six lines (her 13th to 18th)
    for k in range(6):
        warm("mom", "mom", 502.0 + 12.0 * k, 13 + k)
    chat(486.0, "mom")
    lines.sort(key=lambda ln: ln.t)
    visits = (
        Visit("guest", 30.0, 100.0),
        Visit("mom", 125.0, 306.0),
        Visit("dad", 330.0, 450.0),
        Visit("mom", 470.0, 575.0),
    )
    return Script(
        "day",
        DAY_S,
        DAY_MINUTE,
        False,
        visits,
        tuple(lines),
        tuple((t, t + 4.0) for t in touches),
        bang_t=396.0,
    )


# ── the world: SynthWorld, and the family the script brings ─────────────────────────────


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class ConvWorld:
    """SynthWorld with its own family switched off and the script's people in it."""

    def __init__(self, script: Script, seed: int) -> None:
        self.script = script
        self.world = SynthWorld(seed=seed, family=False, day_minute0=script.day_minute0)
        self._rng = random.Random(f"look:{seed}")
        self._pos: list[float] | None = None
        self._who: str | None = None
        self._track = 0
        self._next_track = 0
        self._seen_t = -1e9
        self._speed = 0.0
        if script.dark:
            self.world.lights.append((0, 0.04))
        for ln in script.lines:
            self._sounds(ln)
        if script.bang_t is not None:
            self.world.sounds.append(
                SoundEvent(int(round(script.bang_t * 1000)), "bang", 92.0, 200, dist=VIEW_RANGE_M)
            )
        for t0, t1 in script.touches:
            self.world.touches.append(
                (int(round(t0 * 1000)), int(round(t1 * 1000)), "head", 0.35, 0.6)
            )

    def _sounds(self, ln: Line) -> None:
        """The line as the duck's microphone gets it: its first word's kind, then plain voice."""
        dist = 2.0 if not ln.who else VIEW_RANGE_M
        t0 = int(round(ln.t * 1000))
        first = {"name": 700, "hello": 600}.get(ln.kind, 0)
        total = int(round(ln.dur * 1000))
        if first:
            self.world.sounds.append(SoundEvent(t0, ln.kind, 66.0, first, dist=dist))
        self.world.sounds.append(SoundEvent(t0 + first, "voice", 64.0, total - first, dist=dist))

    # the person who is here now, and where they stand
    def _visit(self, t: float) -> Visit | None:
        for v in self.script.visits:
            if v.t_in <= t < v.t_out:
                return v
        return None

    def _axis_point(self, dist: float) -> tuple[float, float]:
        w = self.world
        a = w.yaw + w._head_yaw
        return (
            min(5.7, max(0.3, w.x + dist * math.cos(a))),
            min(5.7, max(0.3, w.y + dist * math.sin(a))),
        )

    def _in_front_now(self, t: float, visit: Visit) -> bool:
        """Whether the person is stepping in front of the duck's camera axis now: for a line to the
        duck (the 1.6 s before it), to be introduced (from the 1.6 s before the line until the
        proposal it makes has come, 2.5 s after it ends: they stand there to be introduced), and
        for the bang (the 2 s before it: a person who startles the duck does it from in front)."""
        for ln in self.script.lines:
            if ln.who == visit.who and ln.addressed and 0.0 <= ln.t - t <= 1.6:
                return True
            if ln.group == "intro" and -1.6 <= t - ln.t <= ln.dur + 2.5:
                return True
        bang = self.script.bang_t
        if bang is not None and visit.who == "dad" and -0.5 <= t - bang + 2.0 <= 2.5:
            return True
        # (a person the duck is not looking at steps round in front of it, as one talking to a duck does)
        return t - max(self._seen_t, visit.t_in) > UNSEEN_RESET_S

    def _step_person(self, t: float, dt: float) -> None:
        visit = self._visit(t)
        if visit is None:
            self._pos, self._who, self._speed = None, None, 0.0
            return
        if self._pos is None or self._who != visit.who:
            self._who = visit.who
            self._pos = list(self._axis_point(VIEW_RANGE_M + 0.8))
        px, py = self._pos
        far = math.hypot(px - self.world.x, py - self.world.y) > RE_APPROACH_M
        coming = self._in_front_now(t, visit)
        self._speed = 0.0
        if far or coming:
            gx, gy = self._axis_point(VIEW_RANGE_M)
            d = math.hypot(gx - px, gy - py)
            if d > 0.05:
                step = min(d, WALK_M_S * dt)
                self._pos = [px + (gx - px) / d * step, py + (gy - py) / d * step]
                self._speed = step / max(dt, 1e-6)

    def frame(self, t_ms: int) -> C.SensorFrame:
        w = self.world
        t = t_ms / 1000.0
        frame = w.frame(t_ms)
        self._step_person(t, 0.1)
        cam = frame.camera
        if self._pos is None or not cam.usable or cam.light == "black":
            return frame  # (nobody, or the lens sees nothing in the dark)
        view = w.yaw + w._head_yaw
        sight = w._sight(self._pos[0], self._pos[1], view, "person", 1, 0.35)
        if sight is None:
            return frame
        if t - self._seen_t > 0.8 or self._track == 0:  # (the eye re-issues a track after a gap)
            self._next_track += 1
            self._track = 1000 + self._next_track
        self._seen_t = t
        base = LOOKS[self._who or "guest"]
        look = tuple(
            round(v + self._rng.gauss(0.0, LOOK_NOISE * s), 4)
            for v, s in zip(base, LOOK_SCALE, strict=True)
        )
        seen = C.Sighting(
            "person",
            sight[0],
            0.3,
            sight[2],
            0.25,
            min(1.0, 2.0 * self._speed),
            0.0,
            self._track,
            "model",
            look=look,
        )
        return replace(frame, camera=replace(cam, sightings=(*cam.sightings, seen)))

    def apply(self, cmd: C.MotorCommand, dt: float) -> None:
        self.world.apply(cmd, dt)


# ── the cloud: the fake Gemini's proposals, as calls the session would have queued ──────

CLEAN = CallMark(False, 0.9, 0.0, False, False)


def queued(seq: int, call: dict[str, Any], t: float) -> QueuedCall:
    parsed = parse_call(str(call["name"]), call["args"])
    return QueuedCall(
        seq,
        str(call["id"]),
        parsed.name,
        parsed.args,
        parsed.raw,
        parsed.problems,
        parsed.off_bank,
        t,
        0.0,
        1,
        CLEAN,
        False,
    )


def proposals_for(script: Script, seed: int) -> list[tuple[float, int, int, QueuedCall]]:
    """Every call the fake makes to the script's lines: ``(arrival, seq, line index, call)``."""
    fake = FakeGemini(duck_name=NAME, tag_counts=TAG_COUNTS, chat_seed=seed)
    out: list[tuple[float, int, int, QueuedCall]] = []
    seq = 0
    for index, ln in enumerate(script.lines):
        if ln.group == "chat":
            continue
        arrive = ln.end + TEXT_DELAY_S + REPLY_DELAY_S
        for call in fake.react(ln.text):
            seq += 1
            out.append((arrive, seq, index, queued(seq, call, arrive)))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


# ── the run ────────────────────────────────────────────────────────────────────────────


@dataclass
class DayRun:
    preset: str
    seed: int
    kind: str
    lines: list[dict[str, Any]] = field(default_factory=list)  # addressed lines
    proposals: list[dict[str, Any]] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    gate_refused: dict[str, int] = field(default_factory=dict)
    call_out_starts: int = 0
    bond: dict[str, Any] = field(default_factory=dict)
    drive_counts: dict[str, int] = field(default_factory=dict)
    bridge_counts: dict[str, int] = field(default_factory=dict)
    behaviours: dict[str, float] = field(default_factory=dict)  # seconds per behaviour
    people: list[dict[str, Any]] = field(default_factory=list)
    faults: int = 0
    ticks: int = 0
    trace: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "trace" or v is not None}


def _unavailable(asleep: bool, fear: float, reflex: str | None, behaviour: str) -> bool:
    return asleep or fear >= FEAR_MUTE or reflex in FRIGHT_REFLEXES or behaviour == "flee"


def run_day(
    preset: str,
    seed: int,
    params: DriveParams | None = None,
    *,
    kind: str = "day",
    keep_trace: bool = False,
    bridge_config: BridgeConfig | None = None,
    attach: bool = True,
) -> DayRun:
    """Live one scripted day (or night) as ``preset`` with ``seed``. ``attach=False`` runs the same
    day with no bridge at all (the reference for "nothing changes when no one talks")."""
    from duckmind.neuro.integration import install_default_skills, make_brain
    from duckmind.neuro.select import SLEEP
    from duckmind.verify.temperaments import genome

    install_default_skills()
    script = make_script(seed, kind)
    brain = make_brain(genome(preset, seed), seed)
    clock = _Clock()
    bridge: LiveBridge | None = None
    if attach:
        cfg = bridge_config or BridgeConfig(duck_name=NAME, drives=params or DriveParams())
        bridge = LiveBridge(None, config=cfg, clock=clock).attach(brain)
    if kind == "night":
        brain.drives.sleep_pressure = 0.95
    world = ConvWorld(script, seed)
    todo = proposals_for(script, seed)
    seq_line = {seq: index for _t, seq, index, _c in todo}
    result = DayRun(preset, seed, kind, trace=[] if keep_trace else None)
    dt = 1.0 / HZ
    n = int(round(script.length_s * HZ))
    avail: list[bool] = []
    reflexes: list[str | None] = []
    in_view: list[bool] = []
    bond_mom: list[float] = []
    bond_dad: list[float] = []
    prev_bout = -1
    prev_beh = ""
    call_starts: list[float] = []
    speech_spans = [(ln.t, ln.end) for ln in script.lines]
    pending = list(todo)
    time_in: Counter[str] = Counter()
    for k in range(n):
        t = round(k * dt, 6)
        clock.t = t
        while pending and pending[0][0] <= t + 1e-9:
            _arrive, _seq, _index, call = pending.pop(0)
            if bridge is not None:
                bridge.submit(call)
        t_ms = int(round(t * 1000))
        frame = world.frame(t_ms)
        cmd, _tel = brain.step(t_ms, frame)
        world.apply(cmd, dt)
        beh = str(cmd.behaviour)
        asleep = bool(cmd.asleep) or beh in SLEEP
        fear = float(brain.drives.fear)
        reflex = None if cmd.reflex is None else str(cmd.reflex)
        avail.append(not _unavailable(asleep, fear, reflex, beh))
        reflexes.append(reflex)
        seen = bridge is not None and bool(bridge.perception.people)
        in_view.append(seen)
        mom = brain.learn.person_named("媽媽")
        dad = brain.learn.person_named("爸爸")
        bond_mom.append(0.0 if mom is None else float(mom.bond))
        bond_dad.append(0.0 if dad is None else float(dad.bond))
        time_in[beh] += 1
        new_bout = beh != prev_beh or cmd.bout != prev_bout
        if beh == "call_out" and new_bout and seen:
            heard = any(a - 0.05 <= t and t <= b + 6.0 for a, b in speech_spans)
            if not heard:
                call_starts.append(t)
        prev_beh, prev_bout = beh, cmd.bout
        if keep_trace and result.trace is not None:
            result.trace.append(
                {
                    "t": t,
                    "beh": beh,
                    "fear": round(fear, 3),
                    "reflex": reflex,
                    "vocal": cmd.vocal,
                    "seen": seen,
                    "asleep": asleep,
                }
            )
    result.ticks = n
    result.call_out_starts = len(call_starts)
    result.behaviours = {b: round(c * dt, 2) for b, c in sorted(time_in.items())}
    if bridge is None:
        return result
    result.faults = bridge.faults + (0 if brain.live is bridge else 1)
    result.bridge_counts = dict(sorted(bridge.counts.items()))
    result.gate_refused = dict(sorted(bridge.gate.refusals.items()))
    if bridge.drives is not None:
        result.drive_counts = dict(sorted(bridge.drives.counts.items()))
    result.people = brain.learn.people_view()
    _collect(result, bridge, script, seq_line, avail, reflexes, dt)
    _bond_facts(result, script, bond_mom, bond_dad, reflexes, dt)
    return result


def _tick(t: float, dt: float, size: int) -> int:
    return max(0, min(size - 1, int(round(t / dt))))


def _collect(
    result: DayRun,
    bridge: LiveBridge,
    script: Script,
    seq_line: dict[int, int],
    avail: list[bool],
    reflexes: list[str | None],
    dt: float,
) -> None:
    lines = script.lines
    voiced_by: dict[int, list[dict[str, Any]]] = {}
    for v in bridge.voiced:
        voiced_by.setdefault(int(v["intent"]), []).append(v)
        if not avail[_tick(float(v["t"]), dt, len(avail))]:
            result.violations.append({"t": v["t"], "tag": v["tag"], "intent": v["intent"]})
    for entry in bridge.entered:
        if entry.get("tool") != "vocalize":
            continue
        index = seq_line.get(int(entry["seq"]))
        if index is None:
            continue
        ln = lines[index]
        tags = voiced_by.get(int(entry["intent"]), [])
        t_enter = float(entry["t"])
        first = min((float(v["t"]) for v in tags), default=None)
        result.proposals.append(
            {
                "group": ln.group,
                "who": ln.who,
                "ordinal": ln.ordinal,
                "line_t": ln.t,
                "line_end": ln.end,
                "t_enter": t_enter,
                "available": bool(avail[_tick(t_enter, dt, len(avail))]),
                "voiced": bool(tags),
                "latency": None if first is None else round(first - ln.end, 3),
                "tags_voiced": len(tags),
                "tags_proposed": len(entry.get("tags", [])),
            }
        )
    for ln in lines:
        if ln.addressed:
            result.lines.append(
                {"group": ln.group, "who": ln.who, "ordinal": ln.ordinal, "t": ln.t, "end": ln.end}
            )


def _bond_facts(
    result: DayRun,
    script: Script,
    bond_mom: list[float],
    bond_dad: list[float],
    reflexes: list[str | None],
    dt: float,
) -> None:
    if script.kind != "day":
        return
    size = len(bond_mom)
    mom_first = min(ln.t for ln in script.lines if ln.group == "mom")
    mom_last_visit_end = script.visits[-1].t_out
    bang = script.bang_t
    facts: dict[str, Any] = {
        "mom_before": bond_mom[_tick(mom_first - 0.1, dt, size)],
        "mom_end": bond_mom[_tick(mom_last_visit_end, dt, size)],
    }
    if bang is not None:
        window = reflexes[_tick(bang, dt, size) : _tick(bang + 1.5, dt, size) + 1]
        facts["startled"] = any(r in ("startle", "withdraw") for r in window)
        facts["dad_before"] = bond_dad[_tick(bang - 0.1, dt, size)]
        facts["dad_after"] = bond_dad[_tick(bang + 2.5, dt, size)]
    result.bond = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in facts.items()}


# ── the targets, as the frozen file words them ──────────────────────────────────────────


def _rate(items: Sequence[dict[str, Any]]) -> float | None:
    return None if not items else sum(1 for p in items if p["voiced"]) / len(items)


def _avail(proposals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in proposals if p["available"]]


@dataclass
class Pooled:
    """One preset's runs, pooled: the numbers the targets are made of."""

    preset: str
    runs: list[dict[str, Any]]
    nights: list[dict[str, Any]]

    def proposals(self) -> list[dict[str, Any]]:
        return [p for r in self.runs for p in r["proposals"]]

    def available(self) -> list[dict[str, Any]]:
        return _avail(self.proposals())

    def numbers(self) -> dict[str, Any]:
        avail = self.available()
        voiced = [p for p in avail if p["voiced"]]
        stranger = [p for p in avail if p["group"] == "guest"]
        warm = [p for p in avail if p["group"] == "mom" and p["ordinal"] >= 6]
        latencies = [p["latency"] for p in voiced if p["latency"] is not None]
        tags = sum(p["tags_voiced"] for p in voiced)
        per_run = [sum(1 for p in _avail(r["proposals"]) if p["voiced"]) for r in self.runs]
        visitors = {
            who: any(p["voiced"] for p in avail if p["who"] == who)
            for who in ("guest", "mom", "dad")
        }
        night = [p for r in self.nights for p in r["proposals"]]
        unavailable = [p for p in self.proposals() if not p["available"]] + [
            p for p in night if not p["available"]
        ]
        violations = sum(len(r["violations"]) for r in [*self.runs, *self.nights])
        starts = [r["call_out_starts"] for r in self.runs]
        rises = [r["bond"]["mom_end"] - r["bond"]["mom_before"] for r in self.runs if r.get("bond")]
        startled = [r for r in self.runs if r.get("bond", {}).get("startled")]
        drops = [(r["bond"]["dad_before"], r["bond"]["dad_after"]) for r in startled]
        return {
            "runs": len(self.runs),
            "proposals": len(self.proposals()),
            "available": len(avail),
            "rate_all": _rate(avail),
            "rate_stranger": _rate(stranger),
            "n_stranger": len(stranger),
            "rate_warm": _rate(warm),
            "n_warm": len(warm),
            "latency_median": statistics.median(latencies) if latencies else None,
            "tags_per_reply": tags / len(voiced) if voiced else None,
            "voiced": len(voiced),
            "runs_with_a_reply": sum(1 for c in per_run if c >= 1),
            "visitors_answered": visitors,
            "unavailable_proposals": len(unavailable),
            "violations": violations,
            "call_out_starts": starts,
            "call_out_mean": statistics.fmean(starts) if starts else None,
            "bond_rises": rises,
            "startled_runs": len(startled),
            "bond_drops": drops,
        }


def pool(results: Iterable[dict[str, Any]]) -> dict[str, Pooled]:
    by: dict[str, Pooled] = {}
    for r in results:
        p = by.setdefault(r["preset"], Pooled(r["preset"], [], []))
        (p.runs if r["kind"] == "day" else p.nights).append(r)
    return by


T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
}  # fmt: skip


def _t975(df: int) -> float:
    """The two-sided 95 % critical value of Student's t (the nearest tabulated degrees of freedom
    below ``df``; 1.96 beyond 30)."""
    if df > 30:
        return 1.96
    return T975[max(k for k in T975 if k <= max(1, df))]


def _paired_ci(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """The mean of a − b and its 95 % interval, paired by seed."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    m = statistics.fmean(diffs)
    if len(diffs) < 2:
        return m, m, m
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    half = _t975(len(diffs) - 1) * se
    return m, m - half, m + half


@dataclass
class Verdict:
    name: str
    passed: bool
    detail: dict[str, Any]


def targets_report(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The seven targets of ``LV4-TARGETS.md`` on the runs given (day runs and night runs of the four
    presets): PASS/FAIL each, with the numbers."""
    pooled = pool(results)
    nums = {name: pooled[name].numbers() for name in PRESETS if name in pooled}
    out: list[Verdict] = []

    def n(name: str) -> dict[str, Any]:
        return nums[name]

    def ok(x: float | None, test: Callable[[float], bool]) -> bool:
        return x is not None and test(x)

    # 1 songbird
    song = n("social_songbird")
    others = [p for p in PRESETS if p != "social_songbird"]
    rate_top = ok(song["rate_all"], lambda r: r >= 0.8) and all(
        (n(o)["rate_all"] or 0.0) < song["rate_all"] for o in others
    )
    means = {p: n(p)["call_out_mean"] or 0.0 for p in PRESETS}
    runner = max(others, key=lambda p: means[p])
    seeds_song = {r["seed"]: r["call_out_starts"] for r in pooled["social_songbird"].runs}
    seeds_run = {r["seed"]: r["call_out_starts"] for r in pooled[runner].runs}
    common = sorted(set(seeds_song) & set(seeds_run))
    diff, lo, hi = _paired_ci([seeds_song[s] for s in common], [seeds_run[s] for s in common])
    call_top = means["social_songbird"] > means[runner] and lo > 0.0
    out.append(
        Verdict(
            "1 social_songbird talks most",
            bool(rate_top and call_top),
            {
                "rate_all": song["rate_all"],
                "rates": {p: n(p)["rate_all"] for p in PRESETS},
                "call_out_mean": means,
                "runner_up": runner,
                "paired_diff": [round(diff, 3), round(lo, 3), round(hi, 3)],
            },
        )
    )
    # 2, 3 snuggler
    snug = n("shy_snuggler")
    out.append(
        Verdict(
            "2 shy_snuggler: a stranger",
            ok(snug["rate_stranger"], lambda r: r <= 0.3),
            {"rate": snug["rate_stranger"], "n": snug["n_stranger"]},
        )
    )
    out.append(
        Verdict(
            "3 shy_snuggler: warmed up",
            ok(snug["rate_warm"], lambda r: r >= 0.7),
            {"rate": snug["rate_warm"], "n": snug["n_warm"]},
        )
    )
    # 4 dreamer
    dream = n("lazy_dreamer")
    lat = {p: n(p)["latency_median"] for p in PRESETS}
    later = lat["lazy_dreamer"] is not None and all(
        lat[o] is not None and lat["lazy_dreamer"] > lat[o] for o in PRESETS if o != "lazy_dreamer"
    )
    out.append(
        Verdict(
            "4 lazy_dreamer rarely and late",
            bool(ok(dream["rate_all"], lambda r: 0.2 <= r <= 0.5) and later),
            {"rate_all": dream["rate_all"], "latency_median": lat},
        )
    )
    # 5 explorer
    exp = n("bold_explorer")
    out.append(
        Verdict(
            "5 bold_explorer sometimes, briefly",
            bool(
                ok(exp["rate_all"], lambda r: 0.2 <= r <= 0.5)
                and ok(exp["tags_per_reply"], lambda x: x <= 1.3)
                and exp["runs_with_a_reply"] == exp["runs"]
                and all(exp["visitors_answered"].values())
            ),
            {
                "rate_all": exp["rate_all"],
                "tags_per_reply": exp["tags_per_reply"],
                "runs_with_a_reply": f"{exp['runs_with_a_reply']}/{exp['runs']}",
                "visitors_answered": exp["visitors_answered"],
            },
        )
    )
    # 6 no reply while frightened or asleep
    quiet = {p: (n(p)["violations"], n(p)["unavailable_proposals"]) for p in PRESETS}
    out.append(
        Verdict(
            "6 no reply while frightened or asleep",
            all(v == 0 and u >= 8 for v, u in quiet.values()),
            {"violations_and_unavailable_proposals": quiet},
        )
    )
    # 7 bond
    bond_ok = True
    bond_detail: dict[str, Any] = {}
    for p in PRESETS:
        nm = n(p)
        rises = nm["bond_rises"]
        drops = nm["bond_drops"]
        rise_ok = bool(rises) and all(r >= 0.20 for r in rises)
        drop_ok = nm["startled_runs"] >= 12 * len(nm["bond_rises"]) / 16 and all(
            after <= before - 0.03 and after >= 0.2 * before for before, after in drops
        )
        bond_ok = bond_ok and rise_ok and drop_ok
        bond_detail[p] = {
            "rise_min": min(rises) if rises else None,
            "rise_ok": rise_ok,
            "startled_runs": f"{nm['startled_runs']}/{nm['runs']}",
            "drop_ok": drop_ok,
        }
    out.append(Verdict("7 bond rises and drops", bond_ok, bond_detail))
    return {
        "verdicts": [{"name": v.name, "pass": v.passed, "detail": v.detail} for v in out],
        "numbers": nums,
    }


# ── the search's objective ──────────────────────────────────────────────────────────────


def search_loss(
    results: Iterable[dict[str, Any]], margin: float = 0.05
) -> tuple[float, dict[str, float]]:
    """A sum of squared hinge violations of the targets, each taken ``margin`` inside its threshold
    (search design, not part of the targets). Structural target 6 is not in it. Returns the loss and
    each term."""
    pooled = pool(results)
    nums = {name: pooled[name].numbers() for name in PRESETS if name in pooled}
    terms: dict[str, float] = {}

    def hinge(
        name: str, value: float | None, low: float | None = None, high: float | None = None
    ) -> None:
        v = 0.0
        if value is None:
            v = 1.0
        else:
            if low is not None and value < low + margin:
                v += (low + margin - value) ** 2
            if high is not None and value > high - margin:
                v += (value - (high - margin)) ** 2
        terms[name] = v

    song, snug = nums["social_songbird"], nums["shy_snuggler"]
    dream, exp = nums["lazy_dreamer"], nums["bold_explorer"]
    hinge("song_rate", song["rate_all"], low=0.8)
    others_top = max((nums[p]["rate_all"] or 0.0) for p in PRESETS if p != "social_songbird")
    hinge("song_top_rate", (song["rate_all"] or 0.0) - others_top, low=0.0)
    means = {p: nums[p]["call_out_mean"] or 0.0 for p in PRESETS}
    second = max(v for p, v in means.items() if p != "social_songbird")
    terms["song_call_out"] = 0.25 * max(0.0, 1.0 - (means["social_songbird"] - second)) ** 2
    hinge("snug_stranger", snug["rate_stranger"], high=0.3)
    hinge("snug_warm", snug["rate_warm"], low=0.7)
    hinge("dream_rate", dream["rate_all"], low=0.2, high=0.5)
    lat = {p: nums[p]["latency_median"] for p in PRESETS}
    if lat["lazy_dreamer"] is None:
        terms["dream_late"] = 1.0
    else:
        rest = [v for p, v in lat.items() if p != "lazy_dreamer" and v is not None]
        gap = lat["lazy_dreamer"] - (max(rest) if rest else 0.0)
        terms["dream_late"] = max(0.0, 0.3 - gap) ** 2 if gap < 0.3 else 0.0
    hinge("exp_rate", exp["rate_all"], low=0.2, high=0.5)
    hinge("exp_tags", exp["tags_per_reply"], high=1.3)
    silent = exp["runs"] - exp["runs_with_a_reply"]
    terms["exp_never_ignores"] = (silent / max(1, exp["runs"])) ** 2 + sum(
        0.1 for ok_ in exp["visitors_answered"].values() if not ok_
    )
    bond = 0.0
    for p in PRESETS:
        rises = nums[p]["bond_rises"]
        bond += sum(max(0.0, 0.25 - r) ** 2 for r in rises) / max(1, len(rises))
        drops = nums[p]["bond_drops"]
        bond += sum(
            max(0.0, after - (before - 0.05)) ** 2 + max(0.0, 0.25 * before - after) ** 2
            for before, after in drops
        ) / max(1, len(drops))
        bond += (1.0 - nums[p]["startled_runs"] / max(1, nums[p]["runs"])) * 0.0
    terms["bond"] = bond
    return sum(terms.values()), terms


__all__ = [
    "HELD_OUT_SEEDS",
    "NAME",
    "PRESETS",
    "TRAIN_SEEDS",
    "ConvWorld",
    "DayRun",
    "Line",
    "Pooled",
    "Script",
    "Verdict",
    "Visit",
    "make_script",
    "pool",
    "proposals_for",
    "run_day",
    "search_loss",
    "targets_report",
]
