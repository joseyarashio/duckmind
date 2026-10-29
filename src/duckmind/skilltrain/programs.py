"""Trial programs for the Stage-1 gates: one duck, one posture, one skill (or one v5 baseline).

A trial: a fresh :class:`~duckmind.skilltrain.skillduck.SkillDuck` (one DR draw) boots seated
as the owner's simulator does and rises through sitstand; then the posture — **stand** (settle
1.0–1.5 s), **walk** (the twist for 3.0–3.5 s: a steady gait, the phase at the skill's start
varying with the draw), or **sit** (``sit_toggle``, the ~2 s glide, 3.0–3.5 s settled); then at
``t_req`` the client sends ``robot.do <skill>`` exactly once (or re-requests it every 0.1 s for
``hold_s`` and then sends ``then``) and, like the neural duck freezing, sends a zero twist while
the skill runs; when robotd hands back, the post phase: standing still, walking resumed at the
same speed, or seated, for 5 s; a duck that fell is followed until it has been upright for 1 s,
or for 8 s after the fall.

**v5 baselines** (``kind = "v5"``): what the duck does today in the same place, run through the
same harness, the same draws and the same metrics — the "before" of every gate:

========================  =====  ==========================================================
``head_startle``          stand  neck pull-back (neck +0.3, head −0.2) for 0.5 s, the
                                 neural duck's startle head program
``stop``                  walk   the twist to 0 for 1.3 s (the freeze today), then resumed
``back_step``             stand  vx −0.20 for 1.0 s
``back_burst``            stand  vx −0.40 for 0.6 s
``head_shake``            stand  head yaw ±0.3 rad at 3 Hz for 1.0 s
``look_up``               stand  head pitch −0.6 (look up) for 2.0 s
``look_up``               sit    the same, seated
========================  =====  ==========================================================

The active part is followed by a 0.8 s settle (the unwind's place) before the markers' ``t_end``.
"""

from __future__ import annotations

import math
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from duckmind.skilltrain import dr as DR
from duckmind.skilltrain import metrics as M
from duckmind.skilltrain.skillduck import SkillDuck
from duckmind.skilltrain.table import SkillDef

CONTROL_HZ = 50
RE_REQUEST_S = 0.1  # a held skill is re-requested this often (the chain window is 0.15 s)
SKILL_TIMEOUT_S = 15.0
RIGHTING_FOLLOW_S = 8.0
SETTLE_S = 0.8  # a v5 program's "unwind"


@dataclass(frozen=True)
class Condition:
    key: str
    kind: str  # "skill" | "v5"
    family: str  # the gate family (metrics.visibility)
    posture: str  # stand | walk | sit
    level: float | None = None
    speed: float = 0.0
    skill: str | None = None
    hold_s: float | None = None
    then: str | None = None
    v5: str | None = None
    head_alpha: float = 1.0
    post_s: float = M.HANDBACK_S
    n: int | None = None  # overrides the runner's n (semantics checks)
    note: str = ""
    # (s1b:fix, additive) a second entry requested ``cont_at`` s after the first and then held
    # (a freeze taking over a running startle); ``span``: the markers cover every tick from the
    # first entry's start to the last one's end, gaps included (a crouch → rise → crouch is one
    # trial); ``entry_s``: for the risen / coast postures, the wait before the request
    cont: str | None = None
    cont_at: float = 0.0
    span: bool = False
    entry_s: tuple[float, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


Head = tuple[float, float, float, float]


@dataclass(frozen=True)
class V5Program:
    name: str
    active_s: float
    posture: tuple[str, ...]
    family: str
    twist: tuple[float, float, float] | None = None  # None: the posture's own (0, or the walk)
    head: Callable[[float], Head] | None = None
    note: str = ""


def _const(h: Head) -> Callable[[float], Head]:
    return lambda _t: h


V5_PROGRAMS: dict[str, V5Program] = {
    p.name: p
    for p in (
        V5Program(
            "head_startle",
            0.5,
            ("stand",),
            "startle",
            None,
            _const((0.3, -0.2, 0.0, 0.0)),
            "neck pull-back (neck +0.3, head −0.2), today's startle head program",
        ),
        V5Program(
            "stop",
            1.3,
            ("walk",),
            "startle",
            (0.0, 0.0, 0.0),
            None,
            "the freeze today: twist to 0 through robotd's smoothing",
        ),
        V5Program(
            "back_step",
            1.0,
            ("stand",),
            "hop_back",
            (-0.20, 0.0, 0.0),
            None,
            "a backward twist from standing",
        ),
        V5Program(
            "back_burst",
            0.6,
            ("stand",),
            "hop_back",
            (-0.40, 0.0, 0.0),
            None,
            "a short strong backward burst",
        ),
        V5Program(
            "head_shake",
            1.0,
            ("stand",),
            "shake",
            None,
            lambda t: (0.0, 0.0, 0.3 * math.sin(2 * math.pi * 3.0 * t), 0.0),
            "head yaw ±0.3 rad at 3 Hz (no body channel exists)",
        ),
        V5Program(
            "look_up",
            2.0,
            ("stand", "sit"),
            "stretch",
            None,
            _const((0.0, -0.6, 0.0, 0.0)),
            "look up (head pitch −0.6)",
        ),
    )
}


def seed_of(key: str, rep: int) -> int:
    """The draw's seed: the same condition and rep give the same duck on both servo models."""
    return zlib.crc32(f"{key}/{rep}".encode()) & 0x7FFFFFFF


def _boot_rise(duck: SkillDuck, timeout: float = 10.0) -> float | None:
    """Torque on from the seat (robotd's seated-boot rise) and tick until velstand drives
    (robotd's label leaves ``rise``) and the duck is upright; returns the hand-over time (the
    label change: what the A100's "risen" entries count from), None when it did not stand up
    within ``timeout``."""
    if not duck.released:
        duck.enable()
    start = duck.now()
    handover: float | None = None
    while duck.now() - start < timeout:
        duck.tick()
        if handover is None and duck.policy in ("walk", "stand"):
            handover = duck.now()
        if handover is not None and duck.upright():
            return handover
    return None


def _run(duck: SkillDuck, seconds: float, each: Callable[[float], object] | None = None) -> None:
    for _ in range(max(0, int(round(seconds * CONTROL_HZ)))):
        if each is not None:
            each(duck.now())
        duck.tick()


def run_trial(
    cond: Condition,
    *,
    servo: str,
    rep: int,
    skills: Sequence[SkillDef],
    policies: str | None = None,
    mjcf_dir: str | None = None,
    dr_preset: str | None = None,
    keep_trace: bool = False,
) -> dict[str, Any]:
    """One trial → one row (JSON-safe)."""
    began = time.perf_counter()
    seed = seed_of(cond.key, rep)
    preset = dr_preset or ("recipe" if servo == "bam" else "physical")
    draw = DR.draw(seed, preset)
    rng = np.random.default_rng(seed + 11)
    yaw = float(rng.uniform(-math.pi, math.pi))
    duck = SkillDuck(
        skills,
        servo=servo,
        draw=draw,
        head_alpha=cond.head_alpha,
        keyframe="SIT",
        policies=policies,
        mjcf_dir=mjcf_dir,
        yaw=yaw,
    )
    row: dict[str, Any] = {
        "key": f"{cond.key}/{servo}/{rep}",
        "cond": cond.to_json(),
        "servo": servo,
        "rep": rep,
        "seed": seed,
        "dr": draw.to_json(),
        "duck": duck.info(),
        "excluded": None,
    }

    def excluded(why: str) -> dict[str, Any]:
        row["excluded"] = why
        row["wall_s"] = round(time.perf_counter() - began, 3)
        row["sim_s"] = round(duck.now(), 3)
        return row

    handover: float | None = None
    if cond.posture == "risen":
        # (s1b:fix) the skill right after the seated-boot rise: velstand has just taken over
        # (the A100's "risen" entries, 0.1–1.0 s after the hand-over, where comfort fell 6/384)
        handover = _boot_rise(duck)
        if handover is None:
            return excluded("did not stand up after the seated boot")
    else:
        ok, _ = duck.stand(timeout=10.0, hold=1.0)
        if not ok:
            return excluded("did not stand up after the seated boot")
    # ── the posture ──
    if cond.posture == "stand":
        _run(duck, float(rng.uniform(1.0, 1.5)), lambda _t: duck.move(0.0))
    elif cond.posture == "risen":
        lo, hi = cond.entry_s or (0.1, 1.0)
        since = duck.now() - float(handover or duck.now())  # (waiting to be upright counts)
        _run(duck, max(0.0, float(rng.uniform(lo, hi)) - since), lambda _t: duck.move(0.0))
        row["risen_after_s"] = round(duck.now() - float(handover or duck.now()), 3)
    elif cond.posture == "coast":
        # (s1b:fix) a walk stopped by the client (robotd's smoothed stop), the skill requested
        # while the trunk still coasts (the A100's "coast" entries) or after the settle
        _run(duck, float(rng.uniform(3.0, 3.5)), lambda _t: duck.move(cond.speed or 0.40))
        lo, hi = cond.entry_s or (0.1, 0.5)
        _run(duck, float(rng.uniform(lo, hi)), lambda _t: duck.move(0.0))
    elif cond.posture == "walk":
        _run(duck, float(rng.uniform(3.0, 3.5)), lambda _t: duck.move(cond.speed))
    elif cond.posture == "sit":
        duck.do("sit_toggle")
        waited = 0.0
        while duck.policy != "sit" and waited < 6.0:
            duck.tick()
            waited += 1.0 / CONTROL_HZ
        if duck.policy != "sit":
            return excluded("did not sit down")
        _run(duck, 3.0 + float(rng.uniform(0.0, 0.5)))
    else:
        raise ValueError(f"posture {cond.posture!r}")
    if duck.safety.fallen or duck.tilt_deg() > 30.0:
        return excluded(f"down before the {cond.kind} (tilt {duck.tilt_deg():.0f}°)")

    walking = cond.posture == "walk"
    t_req = duck.now()
    t_resume: float | None = None
    acks: list[dict[str, Any]] = []
    if cond.kind == "skill":
        assert cond.skill is not None
        names = (
            [cond.skill] + ([cond.cont] if cond.cont else []) + ([cond.then] if cond.then else [])
        )
        chaining = {d.name for d in skills if d.chain}
        acks.append({"t": round(t_req, 4), "skill": cond.skill, **duck.do(cond.skill)})
        last_req = t_req
        sent_cont = cond.cont is None
        sent_then = False
        t_cont = t_req + cond.cont_at
        t_then = None
        if cond.hold_s is not None:
            t_then = (t_cont if cond.cont else t_req) + cond.hold_s
        deadline = t_req + SKILL_TIMEOUT_S + cond.cont_at + (cond.hold_s or 0.0)
        started = False
        while duck.now() < deadline:
            now = duck.now()
            if cond.posture != "sit":
                duck.move(0.0)
            if not sent_cont and now >= t_cont - 1e-9:
                acks.append({"t": round(now, 4), "skill": cond.cont, **duck.do(str(cond.cont))})
                sent_cont, last_req = True, now
            held = str(cond.cont) if cond.cont and sent_cont else cond.skill
            if t_then is not None and now < t_then - 1e-9:
                # a client holds a chaining entry by re-sending it; a one-shot is just waited on
                if held in chaining and now - last_req >= RE_REQUEST_S - 1e-9:
                    duck.do(held)
                    last_req = now
            elif cond.then and not sent_then and t_then is not None and sent_cont:
                acks.append({"t": round(now, 4), "skill": cond.then, **duck.do(cond.then)})
                sent_then = True
            duck.tick()
            label = str(duck.policy)
            mine = any(label == nm or label == f"{nm}:unwind" for nm in names)
            if mine:
                started = True
            elif started and sent_cont and (not cond.then or sent_then or t_then is None):
                break
        find = M.span_markers if cond.span else M.any_skill_markers
        marks = find(duck.trace.arrays(), names, after=t_req)
        if marks["t_start"] is None:
            return excluded("the skill never started: " + str(acks[-1]))
        if marks["t_end"] is None:
            return excluded("the skill did not hand back within 15 s")
        t_start, t_hold_end, t_end = marks["t_start"], marks["t_hold_end"], marks["t_end"]
        row["sequence"] = marks["sequence"]
    else:
        prog = V5_PROGRAMS[str(cond.v5)]
        t_start = t_req + 1.0 / CONTROL_HZ  # the tick that first carries the new intents

        def during(now: float) -> None:
            rel = now - t_req
            if prog.twist is not None:
                duck.move(*prog.twist)
            elif walking:
                duck.move(cond.speed)
            elif cond.posture == "stand":
                duck.move(0.0)
            if prog.head is not None:
                duck.head(prog.head(rel))

        _run(duck, prog.active_s, during)
        t_hold_end = duck.now()
        duck.head((0.0, 0.0, 0.0, 0.0))
        _run(duck, SETTLE_S, (lambda _t: duck.move(0.0)) if cond.posture != "sit" else None)
        t_end = duck.now()
    # ── the hand-back ──
    if walking:
        t_resume = duck.now()
        _run(duck, cond.post_s, lambda _t: duck.move(cond.speed))
    elif cond.posture in ("stand", "risen", "coast"):
        _run(duck, cond.post_s, lambda _t: duck.move(0.0))
    else:
        _run(duck, cond.post_s)
    # a duck that fell is followed until upright (1 s) or 8 s after the fall
    tr = duck.trace.arrays()
    fallen_idx = np.flatnonzero(tr["fallen"] > 0.5)
    if fallen_idx.size:
        t_fall = float(tr["t"][fallen_idx[0]])
        streak = 0
        while duck.now() < t_fall + RIGHTING_FOLLOW_S + M.UPRIGHT_HOLD_S:
            duck.move(0.0)
            duck.tick()
            streak = streak + 1 if duck.upright() else 0
            if streak >= int(M.UPRIGHT_HOLD_S * CONTROL_HZ) + 1:
                break
        tr = duck.trace.arrays()
    ceiling = float(getattr(duck.servo, "ceiling_nm", 0.64))
    metrics = M.trial_metrics(
        tr,
        t_start=t_start,
        t_hold_end=t_hold_end,
        t_end=t_end,
        t_resume=t_resume,
        ceiling_nm=ceiling,
        post_s=cond.post_s,
    )
    row["markers"] = {
        "t_req": round(t_req, 4),
        "t_start": round(t_start, 4),
        "t_hold_end": round(t_hold_end, 4),
        "t_end": round(t_end, 4),
        "t_resume": None if t_resume is None else round(t_resume, 4),
        "latency_from_request_s": round(t_start - t_req, 4),
    }
    row["acks"] = acks
    row["events"] = [e for e in duck.skill_log if e.get("kind") != "refreshed"][:40]
    row["refreshed"] = sum(1 for e in duck.skill_log if e.get("kind") == "refreshed")
    row["metrics"] = metrics
    judged = "stand" if cond.posture in ("risen", "coast") else cond.posture
    row["visibility"] = M.visibility(
        M.gate_family(cond.family, judged), cond.level, judged, metrics
    )
    row["wall_s"] = round(time.perf_counter() - began, 3)
    row["sim_s"] = round(duck.now(), 3)
    if keep_trace:
        row["trace"] = _compact(tr)
    return row


def _compact(tr: dict[str, Any]) -> dict[str, Any]:
    """Every recorded field (rounded to 5 decimals): enough to re-score a trial with
    :func:`~duckmind.skilltrain.metrics.trial_metrics` without simulating it again."""
    out: dict[str, Any] = {
        k: np.round(np.asarray(v, float), 5).tolist() for k, v in tr.items() if k != "label"
    }
    out["label"] = [str(v) for v in tr["label"]]
    return out


def conditions_for(
    entries: Sequence[Any], *, speeds: Sequence[float] = (0.40, 0.55), head_alpha: float = 1.0
) -> list[Condition]:
    """Every (entry × posture) gate condition of a table (``table.Entry`` objects); walking is
    tried at each of ``speeds`` for the startle family, at 0.40 otherwise."""
    out: list[Condition] = []
    for e in entries:
        for posture in e.postures:
            if posture == "walk":
                fam_speeds = speeds if e.family in ("startle", "freeze", "dummy") else (0.40,)
                for v in fam_speeds:
                    out.append(
                        Condition(
                            key=f"skill/{e.name}/walk{v:.2f}",
                            kind="skill",
                            family=e.family,
                            posture="walk",
                            level=e.level,
                            speed=v,
                            skill=e.name,
                            head_alpha=head_alpha,
                        )
                    )
            else:
                out.append(
                    Condition(
                        key=f"skill/{e.name}/{posture}",
                        kind="skill",
                        family=e.family,
                        posture=posture,
                        level=e.level,
                        skill=e.name,
                        head_alpha=head_alpha,
                    )
                )
    return out


def baseline_conditions(
    *, head_alpha: float = 1.0, speeds: Sequence[float] = (0.40, 0.55)
) -> list[Condition]:
    """The v5 "before" of every Stage-1 gate."""
    out = [
        Condition(
            "v5/head_startle/stand",
            "v5",
            "startle",
            "stand",
            1.0,
            v5="head_startle",
            head_alpha=head_alpha,
        ),
        Condition(
            "v5/back_step/stand",
            "v5",
            "hop_back",
            "stand",
            1.0,
            v5="back_step",
            head_alpha=head_alpha,
        ),
        Condition(
            "v5/back_burst/stand",
            "v5",
            "hop_back",
            "stand",
            1.0,
            v5="back_burst",
            head_alpha=head_alpha,
        ),
        Condition(
            "v5/head_shake/stand",
            "v5",
            "shake",
            "stand",
            1.0,
            v5="head_shake",
            head_alpha=head_alpha,
        ),
        Condition(
            "v5/look_up/stand",
            "v5",
            "stretch_tall",
            "stand",
            1.0,
            v5="look_up",
            head_alpha=head_alpha,
        ),
        Condition(
            "v5/look_up/sit", "v5", "stretch_sit", "sit", 1.0, v5="look_up", head_alpha=head_alpha
        ),
    ]
    out += [
        Condition(
            f"v5/stop/walk{v:.2f}",
            "v5",
            "startle",
            "walk",
            1.0,
            speed=v,
            v5="stop",
            head_alpha=head_alpha,
        )
        for v in speeds
    ]
    return out


def sequence_conditions(entries: Sequence[Any]) -> list[Condition]:
    """Hand-back sequences a trained table must also survive (n as the runner's): a strong
    freeze ended early by ``startle_release``; the deepest crouch cut by ``skill_abort`` after
    0.3 s (velstand takes a crouched duck — the fall-safety path); each from standing and from
    a 0.40 walk. Only the ones whose entries the table has."""
    names = {e.name for e in entries}
    out: list[Condition] = []
    for posture, speed in (("stand", 0.0), ("walk", 0.40)):
        tag = "stand" if posture == "stand" else f"walk{speed:.2f}"
        if {"freeze_4", "startle_release"} <= names:
            out.append(
                Condition(
                    f"seq/freeze_4_release/{tag}",
                    "skill",
                    "freeze",
                    posture,
                    1.0,
                    speed=speed,
                    skill="freeze_4",
                    hold_s=1.0,
                    then="startle_release",
                    note="a 3 s freeze ended after 1 s by the release (its own unwind)",
                )
            )
        if {"startle_4", "skill_abort"} <= names:
            out.append(
                Condition(
                    f"seq/startle_4_abort/{tag}",
                    "skill",
                    "abort",
                    posture,
                    1.0,
                    speed=speed,
                    skill="startle_4",
                    hold_s=0.3,
                    then="skill_abort",
                    note="the deepest crouch handed to velstand mid-hold (no unwind)",
                )
            )
    return out


RISEN_S = (0.1, 1.0)  # the A100's "risen" entries: 0.1–1.0 s after velstand takes over
COAST_S = (0.1, 0.5)  # the client's stop, the trunk still coasting
SETTLED_S = (1.6, 2.2)  # after the guard's settle rule (1.5 s)


def live_conditions(
    entries: Sequence[Any],
    *,
    holds: Sequence[float] = (1.0, 3.0, 10.0),
    walk_hold_s: float = 3.0,
    speeds: Sequence[float] = (0.40, 0.55),
) -> list[Condition]:
    """(s1b:fix) The live table's own conditions, beyond :func:`conditions_for` (which also tries
    each hold entry requested once: a client that stops re-requesting hands a crouched duck to
    velstand after one window):

    * ``hold/<freeze>/<posture>/<H>s`` — the crouch held H s by re-requests, then
      ``startle_release`` (standing: every H; walking: ``walk_hold_s`` at each speed);
    * ``cont/<startle>+<freeze>@<c>/<posture>`` — a one-shot startle taken over by a hold at
      +c s (0.3: inside its hold; 0.9: inside its unwind), held 3 s, released: one crouch, one
      rise (``bob_mm`` over the whole span);
    * ``late/startle_4+freeze_4/<posture>`` — the review's live sequence for comparison: the
      startle hands back, the freeze starts 2.0 s after the request (a crouch → rise → crouch);
    * ``entry/<skill>/<risen|coast>[_settled]`` — comfort skills, hop_back and startle_4 asked
      for 0.1–1.0 s after the seated-boot rise / 0.1–0.5 s after a stop from 0.40 m/s, and the
      same after the guard's 1.5-s settle."""
    by = {e.name: e for e in entries}
    out: list[Condition] = []
    release = "startle_release" if "startle_release" in by else None
    for e in entries:
        if not getattr(e, "hold", False):
            continue
        for h in holds:
            out.append(
                Condition(
                    f"hold/{e.name}/stand/{h:g}s", "skill", "freeze", "stand", e.level,
                    skill=e.name, hold_s=h, then=release, n=None,
                    note=f"held {h:g} s by re-requests, then released",
                )
            )  # fmt: skip
        for v in speeds:
            out.append(
                Condition(
                    f"hold/{e.name}/walk{v:.2f}/{walk_hold_s:g}s", "skill", "freeze", "walk",
                    e.level, speed=v, skill=e.name, hold_s=walk_hold_s, then=release,
                    note=f"from a {v:.2f} walk, held {walk_hold_s:g} s, released",
                )
            )  # fmt: skip
    pairs = [("startle_2", "freeze_3", 0.3), ("startle_4", "freeze_4", 0.3),
             ("startle_4", "freeze_4", 0.9)]  # fmt: skip
    for first, hold, at in pairs:
        if first not in by or hold not in by:
            continue
        for posture, v in (("stand", 0.0), ("walk", 0.40)):
            tag = "stand" if posture == "stand" else f"walk{v:.2f}"
            out.append(
                Condition(
                    f"cont/{first}+{hold}@{at:g}/{tag}", "skill", "continuation", posture,
                    by[first].level, speed=v, skill=first, cont=hold, cont_at=at, hold_s=3.0,
                    then=release, span=True,
                    note=f"{first} taken over by {hold} at +{at:g} s, held 3 s, released",
                )
            )  # fmt: skip
    if {"startle_4", "freeze_4"} <= set(by):
        for posture, v in (("stand", 0.0), ("walk", 0.40)):
            tag = "stand" if posture == "stand" else f"walk{v:.2f}"
            out.append(
                Condition(
                    f"late/startle_4+freeze_4/{tag}", "skill", "continuation", posture, 1.0,
                    speed=v,
                    skill="startle_4", cont="freeze_4", cont_at=2.0, hold_s=3.0, then=release,
                    span=True, note="the review's live sequence: startle, hand-back, freeze",
                )
            )  # fmt: skip
    entry_skills = [n for n in ("shake_1", "shake_2", "stretch_tall", "hop_back", "startle_4")
                    if n in by]  # fmt: skip
    for name in entry_skills:
        e = by[name]
        for posture, window, tag in (
            ("risen", RISEN_S, "risen"),
            ("risen", SETTLED_S, "risen_settled"),
            ("coast", COAST_S, "coast"),
            ("coast", SETTLED_S, "coast_settled"),
        ):
            out.append(
                Condition(
                    f"entry/{name}/{tag}", "skill", e.family, posture, e.level, speed=0.40,
                    skill=name, entry_s=window,
                    note=f"requested {window[0]:g}–{window[1]:g} s after the "
                    + ("seated-boot rise" if posture == "risen" else "stop from a 0.40 walk"),
                )
            )  # fmt: skip
    return out


def semantics_conditions(*, n: int = 5) -> list[Condition]:
    """robotd's skill semantics on the dummy table (small n: these check the path, not the
    body): a chain + unwind entry re-requested for 2 s (robotd never chains it), a chaining entry
    re-requested for 2 s (held), then released by ``skill_abort``; the abort pre-empting a long
    skill; a request while another skill runs (refused)."""
    return [
        Condition(
            "sem/chain_unwind_held2s/stand",
            "skill",
            "dummy",
            "stand",
            skill="dummy_chain_unwind",
            hold_s=2.0,
            n=n,
            note="plan's shape: re-requests are ignored, ends at 1.3 s",
        ),
        Condition(
            "sem/chain_held2s/stand",
            "skill",
            "dummy",
            "stand",
            skill="dummy_chain",
            hold_s=2.0,
            n=n,
            note="chain without unwind: held ≈ 2 s by re-requests",
        ),
        Condition(
            "sem/long_then_abort/stand",
            "skill",
            "dummy",
            "stand",
            skill="dummy_long",
            hold_s=0.6,
            then="skill_abort",
            n=n,
            note="skill_abort pre-empts a 2 s skill after 0.6 s",
        ),
        Condition(
            "sem/long_then_stand/stand",
            "skill",
            "dummy",
            "stand",
            skill="dummy_long",
            hold_s=0.6,
            then="dummy_stand",
            n=n,
            note="a non-chaining request while a skill runs is refused",
        ),
    ]


__all__ = [
    "Condition",
    "V5_PROGRAMS",
    "baseline_conditions",
    "conditions_for",
    "live_conditions",
    "run_trial",
    "seed_of",
    "semantics_conditions",
    "sequence_conditions",
]
