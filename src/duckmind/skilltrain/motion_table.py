"""The skill table the neural duck's live simulator installs (2026-10-05, owner 「把之前已經訓了的動作都接上模擬器」).

One robotd table for every motion trained on Colab, each network the best export by its gate verdict:

* **Stage 1b**, the a3 startle network and the c3 comfort network (the s1b live table: held crouches,
  the finer startle grid). The 2026-09-28 retrain pair a4 / c4 was gated too (``s1b/retrain/gate-a4c4``)
  and is **not** installed: on the owner's XML servos at 20 ms it is no better on any cell that matters
  and worse on three (hop-back 22/30 against 27/30, the DR ducks' shakes 14/30 and 22/30 against 28/30 and
  29/30 on the training servo model); its one gain, the seated stretch's head lift, shows in 0 gate cells;
* **B skills, retrain R1** (``outputs/retrain-20260929``): ``step-b3`` (turn on the spot, march), ``bow-b3``
  (bow, lie low, the rise out of it); R2's ``step-b4`` is worse (march 3/30), ``bow-b4`` is a stand-only
  candidate and not a whole-skill improvement, so R1 stays (R2's own HANDOFF);
* ~~the mat network~~ (job mat-1: step-up fails, step-down passes only on paper) and the page-only ``march`` were taken out of
  the table on 2026-10-09 to free slots for the queue's skills;
* **the hop-over**, R1 ``hop-h3``: the 2 cm and 4 cm height classes (the 1 and 3 cm entries do not fit
  robotd's 30 entries; the network feels the strip, the class only sets the command);
* **the panic gait** (2026-10-09, ``panic_run``; the two arc entries were freed 2026-10-09): v5 fine-tuned to run
  to vx 1.0 (``training/panic-2026-10-08``, R-x90 iteration 1000). Used ONLY for a flight in panic (flee /
  retreat): every other walk stays on v5 velstand. A chaining skill under a constant command (1.0 straight), held by the body link while the flight lasts and handed back to v5 with ``skill_abort``.

robotd takes at most 30 entries and the v5 set's three built-ins count. The table drops the fine-grid entries
``startle_1b``, ``freeze_1b``, ``startle_3b``, ``freeze_2b`` and ``freeze_3b`` (the picker takes the nearest level
that is there) to make room (2026-10-09: ``march``, ``step_up``, ``step_down`` and the two panic arcs freed for the
queue's skills). The official forward roll ``roulade`` is in again (2026-10-26, the flip trick of phase 2: measured in
``ROULADE-2026-10-22.md``): 25 of ours + ``roulade`` + ``kick_left`` / ``kick_right`` = 28 of the 30.

:func:`build` returns the ``skills.json`` document (the s1b live table's shape: ``skills``, ``files`` with
sha256 pins, ``guard`` rules, ``body`` delay); :func:`install` writes it with the networks copied beside it;
:func:`install_defaults` is what ``--brain neuro`` calls at start-up (the table when its files are
the gated ones, else none and the reason; the servo command delay 20 ms either way).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from duckmind import paths
from duckmind.skilltrain import bsk_contract as B
from duckmind.skilltrain import guard as G
from duckmind.skilltrain import hop_contract as H
from duckmind.skilltrain import table as T

DEFAULT_DIR = paths.data_dir() / "live" / "motions"
DEFAULT_TABLE = "skills.json"
NO_TABLE = ("none", "off", "0", "false")
DELAY_ENV = "DUCKMIND_BODY_COMMAND_DELAY_MS"
DEFAULT_DELAY_MS = 20.0  # the servo command delay the skill networks were trained with (15–30 ms)
WALK_DIR = paths.policy_root() / "current"  # the shipped set, whatever $DUCKMIND_POLICY_DIR says
# make room under robotd's 30 entries (the module doc); the last three made room for the panic entries
DROPPED = ("startle_1b", "freeze_1b", "startle_3b", "freeze_2b", "freeze_3b")
# freed 2026-10-09 (planner) for the queue's new skills: the page-only march and the two mat steps (verdict limits / fail)
DROPPED_B = ("march", "step_up", "step_down")
HOP_CLASSES = (2, 4)

# the networks and where they come from (role → file name in the install directory)
FILES: dict[str, str] = {
    "startle": "startle.onnx",
    "comfort": "comfort.onnx",
    "step": "bskstep.onnx",
    "bow": "bskbow.onnx",
    "hop": "hopover.onnx",
    "panic": "panic.onnx",
    "qturn": "quickturn.onnx",
    "roulade": "roulade.onnx",
}
# the official forward roll of the v5 policy set, the one the roll was measured with (``releases/seed-v5``)
ROULADE_SHA256 = "3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b"
ROULADE_DURATION_S = 1.0  # one roll; a chain restarts the window when a request lands in its last 0.15 s
NET_OF = {"bskstep": "step", "bskbow": "bow"}
OPTIONAL = frozenset(
    {"qturn"}
)  # networks of the 2026-10-09 queue: in the table only when a gated file is given
WALK_ROLES = {
    "walk": ("velstand.onnx", "skill_abort runs it; every skill hands back to it"),
    "sitstand": ("alpha_sitstand.onnx", "re-seats the seated stretch"),
}

# the gate verdict of each skill's entries (for the page and the guard): the sources in the module doc
VERDICT: dict[str, dict[str, Any]] = {
    "startle": {"verdict": "pass", "src": "s1b live table, XML servos + 20 ms, n = 30 per cell"},
    "freeze": {"verdict": "pass", "src": "s1b live table (held crouches 1/3/10 s)"},
    "hop_back": {"verdict": "limits", "src": "tr:gate: standing only (from a walk it turns > 15°)"},
    "shake": {"verdict": "pass", "src": "tr:gate / latency map: standing, 15–25 ms"},
    "stretch_tall": {"verdict": "pass", "src": "tr:gate"},
    "stretch_sit": {"verdict": "pass", "src": "tr:gate"},
    "turn": {
        "verdict": "limits",
        "src": "R1 step-b3: standing and walking at 20 ms; weak at 40 ms",
    },
    "bow": {
        "verdict": "pass",
        "src": "R1 bow-b3: standing 28/30 at 20 ms (strain over: caps halved)",
    },
    "lie": {"verdict": "limits", "src": "R1 bow-b3: standing 27/30 at 20 ms, falls at 40 ms"},
    "hop_over": {
        "verdict": "limits",
        "src": "R1 hop-h3: crosses 1–4 cm strips on the XML servos at 20 ms",
    },
    "quick_turn": {
        "verdict": "limits",
        "src": (
            "queue-2026-10-09 turn network (A100 qt-turn-d iteration 1200): fresh ducks n = 60 x servo x 0/20/40 ms: all gates pass "
            "at 0 and 20 ms; at 40 ms on the BAM servos walking 0.5 rad/s moves 5.8 cm per 90 deg (<= 5) and one cell falls 9 (v5 6, limit 8) "
            "(QUICK-TURN-2026-10-09.md)"
        ),
    },
    "roulade": {
        "verdict": "limits",
        "src": (
            "official v5 roulade (ROULADE-2026-10-22.md, robotd --sim + MuJoCo): 30 of 30 open-floor singles upright within 2 s; "
            "from sitting 0 of 10, 0.30 m ahead of a solid 2 of 11 never got up; the roll is 2x over the strain gate (caps: 4 an hour)"
        ),
    },
    "panic": {
        "verdict": "limits",
        "src": (
            "panic-2026-10-08 R-x90-999: G1-G3 and G5 pass on fresh ducks (n = 60), G4 XML at command 0.55 does not "
            "(x1.74): used for the flight only; hand-off H1/H2 (PANIC-2026-10-09.md)"
        ),
    },
}
BODY_NOTE = (
    "the skill networks were trained with training's 15–30 ms servo command delay; on the XML servos "
    "0 falls from 15 to 30 ms, the hop-back performs only at 20–25 ms (s1b latency map)"
)


def sha256_of(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hop_entries(path: str) -> list[T.Entry]:
    out = []
    for v, d in zip(H.variants(), H.skill_defs(path), strict=True):
        if int(round(v.height_m * 100)) not in HOP_CLASSES:
            continue
        out.append(
            T.Entry(
                T.SkillDef.from_json(d),
                "hop_over",
                v.height_m,
                v.postures,
                f"step over a {round(v.height_m * 100)} cm strip (R1 hop-h3)",
            )
        )
    return out


PANIC_DURATION_S = (
    0.4  # the window: a flight not asked for runs on at most this long before robotd hands back
)
PANIC_COMMANDS = (  # name → (vx, vyaw): the gated command (G1: 1.0 straight); the arc entries were freed 2026-10-09
    ("panic_run", (1.0, 0.0, 0.0), 0.0),
)


def _panic_entries(path: str) -> list[T.Entry]:
    out = []
    for name, command, level in PANIC_COMMANDS:
        d = {
            "name": name, "path": path, "duration": PANIC_DURATION_S, "command": command,
            "unwind": (0.0, 0.0, 0.0), "unwind_s": 0.0, "chain": True, "action_scale": 0.9, "gain_ratio": 1.0,
        }  # fmt: skip
        out.append(
            T.Entry(
                T.SkillDef.from_json(d), "panic", level, ("stand", "walk"),
                f"panic flight: vx {command[0]:g}, yaw {command[2]:+g} (the panic network)", hold=True,
            )
        )  # fmt: skip
    return out


QTURN_DURATION_S = (
    0.4  # the window: a turn not asked for runs on at most this long before robotd hands back
)
QTURN_YAW = 2.0  # rad/s: the gated top command (the network turns 1.05-1.3 x of it)
QTURN_COMMANDS = (  # name → (command, level): left is positive yaw
    ("quick_turn_left", (0.0, 0.0, QTURN_YAW), 1.0),
    ("quick_turn_right", (0.0, 0.0, -QTURN_YAW), -1.0),
)


def _qturn_entries(path: str) -> list[T.Entry]:
    out = []
    for name, command, level in QTURN_COMMANDS:
        d = {
            "name": name, "path": path, "duration": QTURN_DURATION_S, "command": command,
            "unwind": (0.0, 0.0, 0.0), "unwind_s": 0.0, "chain": True, "action_scale": 0.9, "gain_ratio": 1.0,
        }  # fmt: skip
        out.append(
            T.Entry(
                T.SkillDef.from_json(d), "quick_turn", level, ("stand", "walk"),
                f"quick turn on the spot: yaw {command[2]:+g} rad/s (the quick-turn network)", hold=True,
            )
        )  # fmt: skip
    return out


def _b_entries(paths: Mapping[str, str]) -> list[T.Entry]:
    out = []
    family = {"turn": "turn", "march": "march", "step_up": "step_up", "step_down": "step_down",
              "bow": "bow", "lie": "lie", "release": "release_bow"}  # fmt: skip
    for e in B.ENTRIES:
        if e.name in DROPPED or e.name in DROPPED_B:
            continue
        d = {
            "name": e.name, "path": paths[NET_OF[e.net]], "duration": e.duration,
            "command": e.command, "unwind": e.unwind, "unwind_s": e.unwind_s, "chain": e.chain,
            "action_scale": B.ACTION_SCALE, "gain_ratio": B.GAIN_RATIO,
        }  # fmt: skip
        out.append(
            T.Entry(
                T.SkillDef.from_json(d),
                family[e.behaviour],
                e.level,
                e.postures,
                e.note,
                hold=e.chain and e.behaviour != "release",
            )
        )
    return out


def _roulade_entry(path: str) -> T.Entry:
    """The official forward roll, at the training values (``action_scale`` / ``gain_ratio`` 1.0: the roll is trained
    against full walking values), a single 1 s window that chains when it is asked again at its end."""
    return T.Entry(
        T.SkillDef(
            "roulade", path, ROULADE_DURATION_S, chain=True, action_scale=1.0, gain_ratio=1.0
        ),
        "roulade",
        None,
        ("stand",),
        "official forward roll (roulade.onnx), one roll about 1 s: standing, open floor, a trick only",
    )


def entries(paths: Mapping[str, str], walk: str) -> list[T.Entry]:
    """Every entry of the table, in robotd's order of appearance: the official roulade, the s1b live table
    without its two shallowest fine-grid entries, the B skills, the hop-over classes."""
    live = [
        e for e in T.stage1_live(paths["startle"], paths["comfort"], walk) if e.name not in DROPPED
    ]
    return [
        _roulade_entry(paths["roulade"]),
        *live,
        *_b_entries(paths),
        *_hop_entries(paths["hop"]),
        *_panic_entries(paths["panic"]),
        *(_qturn_entries(paths["qturn"]) if "qturn" in paths else []),
    ]


def guard_block(names: list[str], base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The guard's rules for these entries: the s1b table's halved caps where it has them
    (``base``: that table's ``guard``), the provisional defaults (this module's family caps) elsewhere."""
    rules: dict[str, Any] = {}
    old = (base or {}).get("rules") or {}
    for name in names:
        r = G.DEFAULT_RULES.get(name)
        if r is None:
            continue
        spec = old.get(name)
        if spec is None:
            spec = {
                "per_hour": r.per_hour,
                "min_gap_s": r.min_gap_s,
                "settle_s": r.settle_s,
                "rise_settle_s": r.rise_settle_s,
                "hold": r.hold,
                "continues": sorted(r.continues),
                "max_walk_mps": r.max_walk_mps,
                "postures": sorted(r.postures),
                "note": r.note,
                "abort_tilt_deg": r.abort_tilt_deg,
                "free_continue": r.free_continue,
                "counted": r.counted,
            }
        rules[name] = spec
    return {"global_per_hour": int((base or {}).get("global_per_hour") or G.GLOBAL_PER_HOUR),
            "rules": rules}  # fmt: skip


def build(
    paths: Mapping[str, str],
    walk: Mapping[str, str],
    *,
    base_guard: Mapping[str, Any] | None = None,
    verdicts: Mapping[str, Any] = VERDICT,
) -> dict[str, Any]:
    """The ``skills.json`` document (``table.FORMAT``). ``paths``: role → absolute ONNX path for
    :data:`FILES`; ``walk``: role → path for the velstand walk network and the sit-stand network
    (pinned: ``skill_abort`` runs the first, every skill hands back to it). Raises ValueError when
    the table is not one robotd takes."""
    walk_path = walk["walk"]
    made = entries(paths, walk_path)
    # the abort entry the stage-1 table carries runs the walk network; B's release entries are theirs
    problems = T.problems(made)
    if problems:
        raise ValueError("; ".join(problems))
    resolved = T.resolve(made)
    files: dict[str, Any] = {}
    for role in FILES:
        if role in OPTIONAL and role not in paths:
            continue
        files[role] = {"path": str(paths[role]), "sha256": sha256_of(paths[role])}
    for role, (_file, why) in WALK_ROLES.items():
        files[role] = {"path": str(walk[role]), "sha256": sha256_of(walk[role]), "role": why}
    skills = []
    for e in made:
        row = e.to_json()
        verdict = verdicts.get(e.family)
        if verdict is not None:
            row["gate"] = {"verdict": verdict["verdict"], "src": verdict["src"],
                           "allowed": list(e.postures)}  # fmt: skip
        skills.append(row)
    names = [e.name for e in made if not e.skill.removed]
    return {
        "format": T.FORMAT,
        "stage": "motions-2026-10-05",
        "table": (
            "every trained motion: s1b live table (a3 startle, c3 comfort), R1 step-b3 / bow-b3, "
            "hop-h3 (2 and 4 cm), the panic gait (flight only, straight), the official roulade (the flip trick); fine-grid entries dropped "
            f"({len(resolved)} entries incl. the v5 kicks)"
        ),
        "files": files,
        "policy": {"none_slots": []},  # (no slot of robotd's is switched off: the roulade entry is the real one)
        "guard": guard_block(names, base_guard),
        "skills": skills,
        "body": {"command_delay_ms": DEFAULT_DELAY_MS, "why": BODY_NOTE},
    }


def install(
    out_dir: str | os.PathLike[str],
    sources: Mapping[str, str | os.PathLike[str]],
    *,
    walk_dir: str | os.PathLike[str] = WALK_DIR,
    base_table: str | os.PathLike[str] | None = None,
) -> Path:
    """Copy the networks (``sources``: role → file) into ``out_dir``, write ``skills.json`` beside
    them (``base_table``: the s1b live ``skills.json`` whose halved caps to keep) and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for role, name in FILES.items():
        if role in OPTIONAL and role not in sources:
            continue
        dest = out / name
        shutil.copyfile(sources[role], dest)
        paths[role] = str(dest.resolve())
    walk_root = Path(walk_dir).resolve()
    walk = {role: str(walk_root / file) for role, (file, _why) in WALK_ROLES.items()}
    base = None
    if base_table is not None:
        base = json.loads(Path(base_table).read_text(encoding="utf-8")).get("guard")
    doc = build(paths, walk, base_guard=base)
    target = out / DEFAULT_TABLE
    target.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def check(path: str | os.PathLike[str]) -> list[str]:
    """What is wrong with an installed table: a missing or changed network (the gated sha256 pins)."""
    from duckmind.skilltrain import gt_wiring as W

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return W.check_pins(doc)


def install_defaults(environ: MutableMapping[str, str] | None = None) -> dict[str, Any]:
    """What ``--brain neuro`` sets before the body starts (never overrides a setting):

    * the servo command delay 20 ms (``DUCKMIND_BODY_COMMAND_DELAY_MS``): the real servos act late and
      the skill networks were trained with it; velstand walks the same with it (v5: 0 falls at every
      delay in the s1b latency map);
    * the trained-motion table (``DUCKMIND_ROBOTD_SKILLS``): ``$DUCKMIND_ROBOTD_SKILLS`` as given (``none``
      switches it off), else ``live/motions/skills.json`` in the data folder when it is there and every
      network is the gated one. Anything else leaves the simulator as it was, and says why.

    Returns ``{"skills": path or None, "delay_ms": ..., "why": reason or None}``."""
    env = os.environ if environ is None else environ
    info: dict[str, Any] = {"skills": None, "delay_ms": None, "why": None}
    if not (env.get(DELAY_ENV) or "").strip():
        env[DELAY_ENV] = f"{DEFAULT_DELAY_MS:g}"
    info["delay_ms"] = env[DELAY_ENV].strip()
    given = (env.get(T.SKILLS_ENV) or "").strip()
    if given.lower() in NO_TABLE:
        env.pop(T.SKILLS_ENV, None)
        info["why"] = f"{T.SKILLS_ENV}={given}"
        return info
    if given:
        info["skills"] = given  # (the owner's own table: gt_wiring checks it strictly at start-up)
        return info
    path = DEFAULT_DIR / DEFAULT_TABLE
    if not path.is_file():
        info["why"] = f"no table at {path}"
        return info
    problems = check(path)
    if problems:
        info["why"] = "; ".join(problems)
        return info
    env[T.SKILLS_ENV] = str(path)
    info["skills"] = str(path)
    return info


__all__ = [
    "DEFAULT_DIR",
    "FILES",
    "VERDICT",
    "build",
    "check",
    "entries",
    "guard_block",
    "install",
    "install_defaults",
]
