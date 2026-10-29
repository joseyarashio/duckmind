"""robotd's one-shot skill table for Stage 1, and the ``robotd.toml`` section that installs it.

**robotd's own rules** (``robotd-params/src/lib.rs`` ``SkillDef`` / ``resolved_skills_with``,
``robotd/src/control.rs``, ``robotd/src/intents.rs``; read-only upstream):

* An entry is ``name, path, duration, command[3], unwind[3], unwind_s, chain`` plus
  ``params.action_scale`` / ``params.gain_ratio``. ``deny_unknown_fields``: any other key makes
  robotd refuse the whole file, so :func:`toml_section` writes exactly these.
* While a skill runs, the head and body commands are **zero** and the twist is the entry's
  constant ``command`` (then ``unwind`` for ``unwind_s``), unsmoothed. Priority: skill > ground
  pick > sit/rise > stand > walk, so a skill pre-empts walking; a seated duck keeps its sit
  state and the sitstand network re-seats it when the skill hands back.
* The table **merges** over the set's built-ins (v5: ``roulade``, ``kick_left``, ``kick_right``,
  in that order): a new name is appended after them, a built-in's name replaces it in place, a
  path of ``"none"`` removes it. At most 30 entries (``MAX_SKILLS``).
  ``ground_pick`` / ``sit_toggle`` are the daemon's own names and may not be used.
* ``chain``: a request landing while the skill runs (within 0.15 s of the window's end)
  restarts it — **but only when ``unwind_s`` is 0**. control.rs checks "Holding and
  ``unwind_s > 0`` → Unwinding" *before* the chain, so a chaining entry with an unwind never
  restarts: re-requests are refreshed and ignored. :func:`problems` flags that combination
  (the training plan's ``startle_k`` rows had it); :func:`stage1` avoids it.
* A chaining entry may **pre-empt** any running skill (any phase); a non-chaining one is refused
  while another skill runs. Two requests in the same tick run in index order, so of two
  chaining ones the higher index wins.

**The Stage-1 table** (:func:`stage1`): the plan's command encodings (training value =
deployment value), one-shot windows with robotd's own unwind, two longer ``freeze_*`` holds for
strong fear, ``startle_release`` (the startle network on its unwind command, chaining: ends a
freeze early with a proper unwind) and ``skill_abort`` (the v5 walk network as a one-tick
chaining skill: hands any skill straight back to velstand, e.g. for righting). Per-skill
``action_scale`` / ``gain_ratio`` are **pinned** (never left to robotd's standing rule).

**The live table** (:func:`stage1_live`, s1b:fix): the same networks and commands, with held
crouches (``freeze_*``: chain, no unwind, re-requested while the fear lasts, ended by
``startle_release``) so a startle that turns into a freeze is one crouch and one rise, and a finer
one-shot grid for per-execution variation. :func:`stage1` stays the training contract.

A table travels as ``skills.json`` (:func:`save` / :func:`load`): robotd's fields plus our
``family`` / ``level`` / ``postures`` / ``hold`` for the gates, which the TOML never carries.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT = "duckmind-skill-table-1"
MAX_SKILLS = 30  # robotd intents.rs: the request mask is a u32 with 2 bits reserved
CHAIN_WINDOW_S = 0.15  # control.rs CHAIN_WINDOW
DAEMON_OWNED = ("ground_pick", "sit_toggle")  # robotd-params DAEMON_OWNED_SKILLS
NONE_SENTINEL = "none"
STANDING_THRESHOLD = 0.05  # duck-control DEFAULT_STANDING_THRESHOLD
# The v5 set's zero-command skills (manifest order), what robotd resolves before our entries.
V5_BUILTINS = (
    ("roulade", "roulade.onnx", 1.0, True),
    ("kick_left", "ball_kick_left.onnx", 0.5, False),
    ("kick_right", "ball_kick_right.onnx", 0.5, False),
)
POSTURES = ("stand", "walk", "sit")
# The plan's command encodings (§2, "training value = deployment value").
STARTLE_LEVELS = (0.25, 0.5, 0.75, 1.0)
HOP_BACK_COMMAND = (1.0, 1.0, 0.0)
SHAKE_LEVELS = (0.5, 1.0)
STRETCH_TALL_COMMAND = (0.0, -1.0, 0.0)
STRETCH_SIT_COMMAND = (0.0, -1.0, 1.0)
ZERO3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SkillDef:
    """One ``[[policy.skill]]`` entry, robotd's fields only."""

    name: str
    path: str | None
    duration: float
    command: tuple[float, float, float] = ZERO3
    unwind: tuple[float, float, float] = ZERO3
    unwind_s: float = 0.0
    chain: bool = False
    action_scale: float | None = None
    gain_ratio: float | None = None

    @property
    def removed(self) -> bool:
        return self.path is not None and self.path.strip().lower() == NONE_SENTINEL

    def toml(self) -> str:
        """This entry as robotd reads it (exactly the ``SkillDef`` keys, nothing else)."""
        lines = ["[[policy.skill]]", f"name = {_toml_str(self.name)}"]
        if self.path is not None:
            lines.append(f"path = {_toml_str(self.path)}")
        lines += [
            f"duration = {_toml_float(self.duration)}",
            f"command = {_toml_vec(self.command)}",
            f"unwind = {_toml_vec(self.unwind)}",
            f"unwind_s = {_toml_float(self.unwind_s)}",
            f"chain = {'true' if self.chain else 'false'}",
        ]
        params = []
        if self.action_scale is not None:
            params.append(f"action_scale = {_toml_float(self.action_scale)}")
        if self.gain_ratio is not None:
            params.append(f"gain_ratio = {_toml_float(self.gain_ratio)}")
        if params:
            lines.append("params = { " + ", ".join(params) + " }")
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "duration": self.duration,
            "command": list(self.command),
            "unwind": list(self.unwind),
            "unwind_s": self.unwind_s,
            "chain": self.chain,
            "action_scale": self.action_scale,
            "gain_ratio": self.gain_ratio,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> SkillDef:
        def vec(value: Any) -> tuple[float, float, float]:
            values = [float(v) for v in (value if value is not None else ZERO3)]
            if len(values) != 3:
                raise ValueError(f"{raw.get('name')!r}: a twist has 3 numbers, not {values}")
            return (values[0], values[1], values[2])

        def opt(value: Any) -> float | None:
            return None if value is None else float(value)

        return cls(
            name=str(raw["name"]),
            path=None if raw.get("path") is None else str(raw["path"]),
            duration=float(raw["duration"]),
            command=vec(raw.get("command")),
            unwind=vec(raw.get("unwind")),
            unwind_s=float(raw.get("unwind_s", 0.0)),
            chain=bool(raw.get("chain", False)),
            action_scale=opt(raw.get("action_scale")),
            gain_ratio=opt(raw.get("gain_ratio")),
        )


@dataclass(frozen=True)
class Entry:
    """A table entry plus what the gates need to know about it (never written to robotd)."""

    skill: SkillDef
    family: str  # startle | freeze | hop_back | shake | stretch_tall | stretch_sit | release |
    #              abort | dummy
    level: float | None = None  # the command's intensity (startle k, shake amplitude)
    postures: tuple[str, ...] = ("stand",)  # where the gates try it (and duck-body may use it)
    note: str = ""
    # (s1b:fix) a held crouch: chaining, no unwind — the client re-requests it while the fear
    # lasts (robotd restarts the window, the network sees the same command: one continuous
    # crouch) and ends it with ``startle_release`` (the network's own rise)
    hold: bool = False

    @property
    def name(self) -> str:
        return self.skill.name

    def to_json(self) -> dict[str, Any]:
        return {
            **self.skill.to_json(),
            "family": self.family,
            "level": self.level,
            "postures": list(self.postures),
            "note": self.note,
            "hold": self.hold,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Entry:
        postures = tuple(str(p) for p in raw.get("postures") or ("stand",))
        bad = [p for p in postures if p not in POSTURES]
        if bad:
            raise ValueError(f"{raw.get('name')!r}: unknown postures {bad}; use {POSTURES}")
        level = raw.get("level")
        return cls(
            skill=SkillDef.from_json(raw),
            family=str(raw.get("family") or "dummy"),
            level=None if level is None else float(level),
            postures=postures,
            note=str(raw.get("note") or ""),
            hold=bool(raw.get("hold", False)),
        )


# ── robotd's rules ────────────────────────────────────────────────────────────────────────────


def problems(entries: Sequence[SkillDef | Entry], *, check_files: bool = True) -> list[str]:
    """Everything robotd would refuse, or would run other than the table's author meant.

    Refusals (robotd would not start with this file): a daemon-owned or empty name, duplicate
    names, a non-finite or non-positive duration, a twist without 3 finite numbers, a negative
    ``unwind_s``, a non-positive scale or gain ratio, a missing/relative/symlinked ONNX, more
    than 30 skills after the v5 built-ins. Meaning traps (it would start, and do something else):
    ``chain`` with ``unwind_s > 0`` (never chains, see the module doc), an unpinned
    ``action_scale`` / ``gain_ratio`` on one of ours (robotd would pick walk or standing tuning
    by the command's magnitude)."""
    found: list[str] = []
    defs = [e.skill if isinstance(e, Entry) else e for e in entries]
    seen: set[str] = set()
    for d in defs:
        where = f"skill {d.name!r}"
        if not d.name.strip():
            found.append("a skill has an empty name")
        if d.name in DAEMON_OWNED:
            found.append(f"{where}: the name belongs to robotd itself ({', '.join(DAEMON_OWNED)})")
        if d.name in seen:
            found.append(f"{where}: listed twice (robotd merges by name: the last one wins)")
        seen.add(d.name)
        if not (math.isfinite(d.duration) and d.duration > 0.0):
            found.append(f"{where}: duration must be a positive number of seconds")
        for label, vec in (("command", d.command), ("unwind", d.unwind)):
            if len(vec) != 3 or not all(math.isfinite(v) for v in vec):
                found.append(f"{where}: {label} must be 3 finite numbers")
        if not (math.isfinite(d.unwind_s) and d.unwind_s >= 0.0):
            found.append(f"{where}: unwind_s must be ≥ 0")
        if d.chain and d.unwind_s > 0.0:
            found.append(
                f"{where}: chain with unwind_s {d.unwind_s:g} never chains in robotd "
                "(control.rs: Holding → Unwinding comes first); re-requests are ignored"
            )
        for label, value in (("action_scale", d.action_scale), ("gain_ratio", d.gain_ratio)):
            if value is not None and not (math.isfinite(value) and value > 0.0):
                found.append(f"{where}: params.{label} must be a positive number")
        if not d.removed and (d.action_scale is None or d.gain_ratio is None):
            found.append(
                f"{where}: pin params.action_scale and params.gain_ratio to the training "
                "values (unpinned, robotd picks walk or standing tuning by |command|)"
            )
        if check_files and not d.removed:
            if d.path is None:
                found.append(f"{where}: give an absolute path (robotd's default is its own dir)")
            else:
                p = Path(d.path)
                if not p.is_absolute():
                    found.append(f"{where}: path must be absolute, not {d.path!r}")
                elif p.is_symlink() or not p.is_file():
                    found.append(f"{where}: {d.path} is not a regular file")
    count = len(resolve(defs))
    if count > MAX_SKILLS:
        found.append(f"{count} skills after the v5 built-ins; robotd takes at most {MAX_SKILLS}")
    return found


def resolve(
    entries: Sequence[SkillDef | Entry], policy_dir: str | os.PathLike[str] | None = None
) -> list[SkillDef]:
    """The skill list robotd ends up with (``resolved_skills_with``): the v5 built-ins in order,
    each configured entry replacing a built-in of the same name in place or appended, and every
    ``"none"`` path removed. The index in this list is robotd's skill index (= priority)."""
    folder = Path(policy_dir) if policy_dir is not None else None
    resolved: list[SkillDef] = [
        SkillDef(
            name=name,
            path=None if folder is None else str(folder / file),
            duration=duration,
            chain=chain,
        )
        for name, file, duration, chain in V5_BUILTINS
    ]
    for item in entries:
        d = item.skill if isinstance(item, Entry) else item
        for i, existing in enumerate(resolved):
            if existing.name == d.name:
                resolved[i] = d
                break
        else:
            resolved.append(d)
    return [d for d in resolved if not d.removed]


def toml_section(entries: Sequence[SkillDef | Entry]) -> str:
    """The ``[[policy.skill]]`` blocks to append to a ``robotd.toml`` whose ``[policy]`` table
    is already written (what our live launcher writes into ``/private/tmp/dm-live-*/``).
    Raises ValueError on anything :func:`problems` would refuse outright."""
    defs = [e.skill if isinstance(e, Entry) else e for e in entries]
    refusals = [p for p in problems(defs) if "never chains" not in p]
    if refusals:
        raise ValueError("robotd would refuse this skill table: " + "; ".join(refusals))
    if not defs:
        return ""
    return "\n# duckmind skilltrain: one-shot skills for this simulator only\n" + "\n".join(
        d.toml() for d in defs
    )


# ── tables ────────────────────────────────────────────────────────────────────────────────────


def stage1(
    startle_onnx: str,
    comfort_onnx: str,
    walk_onnx: str,
    *,
    startle_scale: float = 1.0,
    startle_gain_ratio: float = 1.0,
    comfort_scale: float = 1.0,
    comfort_gain_ratio: float = 1.0,
    walk_scale: float = 0.9,
) -> list[Entry]:
    """The Stage-1 table (training plan §2, fixed for robotd's real chain semantics).

    ``*_scale`` / ``*_gain_ratio`` must be the training env's action scale and kp scale (the
    export manifest carries them). ``walk_scale`` is robotd's walking scale, which velstand runs
    at in the v5 walk slot (``skill_abort`` must feel like velstand, not like a skill)."""

    def startle(name: str, command: tuple[float, float, float], duration: float) -> SkillDef:
        return SkillDef(
            name=name,
            path=startle_onnx,
            duration=duration,
            command=command,
            unwind=ZERO3,
            unwind_s=0.8,
            chain=False,
            action_scale=startle_scale,
            gain_ratio=startle_gain_ratio,
        )

    def comfort(
        name: str, command: tuple[float, float, float], duration: float, unwind_s: float
    ) -> SkillDef:
        return SkillDef(
            name=name,
            path=comfort_onnx,
            duration=duration,
            command=command,
            unwind=ZERO3,
            unwind_s=unwind_s,
            chain=False,
            action_scale=comfort_scale,
            gain_ratio=comfort_gain_ratio,
        )

    walking = ("stand", "walk")
    entries = [
        Entry(
            startle(f"startle_{i + 1}", (k, 0.0, 0.0), 0.5),
            "startle",
            k,
            walking,
            "flinch → crouch, held 0.5 s, robotd's 0.8 s unwind",
        )
        for i, k in enumerate(STARTLE_LEVELS)
    ]
    entries += [
        Entry(
            startle("freeze_3", (0.75, 0.0, 0.0), 2.0),
            "freeze",
            0.75,
            walking,
            "strong fear: the crouch held 2 s",
        ),
        Entry(
            startle("freeze_4", (1.0, 0.0, 0.0), 3.0),
            "freeze",
            1.0,
            walking,
            "strongest fear: the crouch held 3 s",
        ),
        Entry(
            SkillDef(
                "hop_back",
                startle_onnx,
                0.8,
                HOP_BACK_COMMAND,
                ZERO3,
                0.6,
                False,
                startle_scale,
                startle_gain_ratio,
            ),
            "hop_back",
            1.0,
            walking,
            "the largest startle plus 2–3 quick back-steps (D5)",
        ),
    ]
    entries += [
        Entry(
            comfort(f"shake_{i + 1}", (0.0, 0.0, a), 1.0, 0.5),
            "shake",
            a,
            ("stand",),
            "shake-off in place",
        )
        for i, a in enumerate(SHAKE_LEVELS)
    ]
    entries += [
        Entry(
            comfort("stretch_tall", STRETCH_TALL_COMMAND, 2.0, 0.8),
            "stretch_tall",
            1.0,
            ("stand",),
            "standing stretch; duck-body opens the beak (yawn)",
        ),
        Entry(
            comfort("stretch_sit", STRETCH_SIT_COMMAND, 2.0, 0.8),
            "stretch_sit",
            1.0,
            ("sit",),
            "seated half-rise stretch; robotd re-seats it",
        ),
        Entry(
            SkillDef(
                "startle_release",
                startle_onnx,
                0.8,
                ZERO3,
                ZERO3,
                0.0,
                True,
                startle_scale,
                startle_gain_ratio,
            ),
            "release",
            0.0,
            walking,
            "ends a freeze early with the network's own unwind",
        ),
        Entry(
            SkillDef("skill_abort", walk_onnx, 0.02, ZERO3, ZERO3, 0.0, True, walk_scale, 1.0),
            "abort",
            0.0,
            ("stand", "walk", "sit"),
            "hands any running skill straight back to velstand (one tick)",
        ),
    ]
    return entries


# ── the live table (s1b:fix 2026-09-28) ──────────────────────────────────────────────────────

# One-shot startles on a finer grid (per-execution variation: the picker jitters the level and
# takes the nearest entry — the startle network was trained on continuous k between the levels,
# rl_contract.CONTINUOUS_LEVEL_PROB) and held crouches for lasting fear.
LIVE_STARTLE = (
    ("startle_1", 0.25),
    ("startle_1b", 0.375),
    ("startle_2", 0.5),
    ("startle_2b", 0.625),
    ("startle_3", 0.75),
    ("startle_3b", 0.875),
    ("startle_4", 1.0),
)
LIVE_FREEZE = (
    ("freeze_1b", 0.375),
    ("freeze_2", 0.5),
    ("freeze_2b", 0.625),
    ("freeze_3", 0.75),
    ("freeze_3b", 0.875),
    ("freeze_4", 1.0),
)
HOLD_WINDOW_S = 1.0  # a hold's robotd window: re-requested every ≤ 0.15 s before it ends
LIVE_SHAKE_S = 1.0  # the shake's window (the training table's; a longer one only once gated)


def stage1_live(
    startle_onnx: str,
    comfort_onnx: str,
    walk_onnx: str,
    *,
    startle_scale: float = 1.0,
    startle_gain_ratio: float = 1.0,
    comfort_scale: float = 1.0,
    comfort_gain_ratio: float = 1.0,
    walk_scale: float = 0.9,
    shake_s: float = LIVE_SHAKE_S,
) -> list[Entry]:
    """The table the simulator runs (s1b:fix): the training table (:func:`stage1`, unchanged —
    it stays the training contract) with

    * **held crouches** instead of the fixed 2 s / 3 s freezes: ``freeze_*`` chain with no unwind
      (robotd honours the chain only then), a 1 s window the client re-requests while the fear
      lasts — one continuous crouch of any length — ended by ``startle_release`` (the network's
      own rise, the same zero command robotd's unwind gave it). A chaining entry may pre-empt a
      running skill, so a freeze takes over a running startle without a rise in between (the
      review's live crouch → rise → crouch);
    * **a finer one-shot grid** (``startle_1b`` / ``2b`` / ``3b`` between the four levels) and
      holds from ``1b`` (k 0.375) to ``4``: the picker jitters the level per execution;
    * the shake's window ``shake_s`` (1.0 s = the trained one).

    Commands, networks and pinned scale/gain are the training table's."""
    base = {
        e.name: e
        for e in stage1(
            startle_onnx,
            comfort_onnx,
            walk_onnx,
            startle_scale=startle_scale,
            startle_gain_ratio=startle_gain_ratio,
            comfort_scale=comfort_scale,
            comfort_gain_ratio=comfort_gain_ratio,
            walk_scale=walk_scale,
        )
    }
    walking = ("stand", "walk")

    def startle(name: str, k: float) -> Entry:
        return Entry(
            SkillDef(name, startle_onnx, 0.5, (k, 0.0, 0.0), ZERO3, 0.8, False,
                     startle_scale, startle_gain_ratio),
            "startle",
            k,
            walking,
            "flinch → crouch, held 0.5 s, robotd's 0.8 s unwind",
        )  # fmt: skip

    def hold(name: str, k: float) -> Entry:
        return Entry(
            SkillDef(name, startle_onnx, HOLD_WINDOW_S, (k, 0.0, 0.0), ZERO3, 0.0, True,
                     startle_scale, startle_gain_ratio),
            "freeze",
            k,
            walking,
            "fear lasts: the crouch held while re-requested, ended by startle_release",
            hold=True,
        )  # fmt: skip

    out = [startle(n, k) for n, k in LIVE_STARTLE]
    out += [hold(n, k) for n, k in LIVE_FREEZE]
    out.append(base["hop_back"])
    for name in ("shake_1", "shake_2"):
        e = base[name]
        out.append(Entry(
            SkillDef(name, e.skill.path, float(shake_s), e.skill.command, e.skill.unwind,
                     e.skill.unwind_s, False, e.skill.action_scale, e.skill.gain_ratio),
            e.family, e.level, e.postures, e.note,
        ))  # fmt: skip
    out.append(base["stretch_tall"])
    sit = base["stretch_sit"]
    out.append(
        Entry(
            sit.skill,
            sit.family,
            sit.level,
            sit.postures,
            "seated half-rise (+13–14 mm); the head lifts less than a plain seated look-up",
        )
    )
    out.append(base["startle_release"])
    out.append(base["skill_abort"])
    return out


def dummy(walk_onnx: str, *, walk_scale: float = 0.9) -> list[Entry]:
    """Stand-ins that exercise the harness before any trained skill exists: the v5 walk
    network (velstand) run as a skill, on robotd's skill path (zeroed head and body, a
    constant zero twist, pre-empting a walk, re-seated after), in the Stage-1 window shapes.
    ``dummy_chain_unwind`` is the plan's chain + unwind shape (robotd never chains it);
    ``dummy_chain`` chains (unwind_s 0)."""

    def d(name: str, duration: float, unwind_s: float, chain: bool) -> SkillDef:
        return SkillDef(name, walk_onnx, duration, ZERO3, ZERO3, unwind_s, chain, walk_scale, 1.0)

    everywhere = ("stand", "walk", "sit")
    return [
        Entry(
            d("dummy_stand", 0.5, 0.8, False),
            "dummy",
            0.0,
            everywhere,
            "v5 standing as a skill in the startle_k window (0.5 s + 0.8 s unwind)",
        ),
        Entry(
            d("dummy_long", 2.0, 0.8, False),
            "dummy",
            0.0,
            everywhere,
            "v5 standing as a skill in the stretch window (2.0 s + 0.8 s unwind)",
        ),
        Entry(
            d("dummy_chain_unwind", 0.5, 0.8, True),
            "dummy",
            0.0,
            everywhere,
            "the plan's startle shape: chain + unwind (robotd never chains it)",
        ),
        Entry(
            d("dummy_chain", 0.5, 0.0, True),
            "dummy",
            0.0,
            everywhere,
            "chain without unwind: held while re-requested",
        ),
        Entry(
            d("skill_abort", 0.02, 0.0, True),
            "abort",
            0.0,
            everywhere,
            "velstand as a one-tick chaining skill",
        ),
    ]


# ── the live launcher's hook ──────────────────────────────────────────────────────────────────

SKILLS_ENV = "DUCKMIND_ROBOTD_SKILLS"


def robotd_skill_section_from_env(environ: Mapping[str, str] | None = None) -> str:
    """The ``[[policy.skill]]`` section for this simulator's robotd from
    ``$DUCKMIND_ROBOTD_SKILLS`` (a ``skills.json``); empty when unset, so a launcher that appends
    it changes nothing until a table is installed. A bad table raises (robotd would refuse the
    whole file, and a simulator silently without its skills is worse than one that says so)."""
    env = os.environ if environ is None else environ
    raw = (env.get(SKILLS_ENV) or "").strip()
    if not raw:
        return ""
    return toml_section(load(raw))


def skill_names_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The names the installed table adds (for duck-body's allow list); () when unset."""
    env = os.environ if environ is None else environ
    raw = (env.get(SKILLS_ENV) or "").strip()
    return tuple(e.name for e in load(raw) if not e.skill.removed) if raw else ()


# ── files ─────────────────────────────────────────────────────────────────────────────────────


def save(path: str | os.PathLike[str], entries: Iterable[Entry], **meta: Any) -> Path:
    out = Path(path)
    doc = {"format": FORMAT, **meta, "skills": [e.to_json() for e in entries]}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load(path: str | os.PathLike[str]) -> list[Entry]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return [Entry.from_json(raw) for raw in doc.get("skills") or []]


# ── TOML literals (the subset robotd.toml needs) ─────────────────────────────────────────────


def _toml_str(text: str) -> str:
    return json.dumps(str(text), ensure_ascii=False)  # a TOML basic string for these values


def _toml_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"TOML cannot carry {number}")
    return repr(number)


def _toml_vec(values: Sequence[float]) -> str:
    return "[" + ", ".join(_toml_float(v) for v in values) + "]"


__all__ = [
    "CHAIN_WINDOW_S",
    "DAEMON_OWNED",
    "Entry",
    "FORMAT",
    "MAX_SKILLS",
    "POSTURES",
    "SkillDef",
    "V5_BUILTINS",
    "HOLD_WINDOW_S",
    "LIVE_FREEZE",
    "LIVE_STARTLE",
    "dummy",
    "load",
    "problems",
    "resolve",
    "save",
    "SKILLS_ENV",
    "robotd_skill_section_from_env",
    "skill_names_from_env",
    "stage1",
    "stage1_live",
    "toml_section",
]
