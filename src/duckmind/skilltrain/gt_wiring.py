"""tr:gate: what a gated skill table tells our live simulator besides robotd's skill list.

Pure standard library (the neural duck's driver and the live launcher import it; the wiring patch
carries a copy). A ``skills.json`` (:mod:`duckmind.skilltrain.table`) may carry two top-level
blocks that robotd never sees:

``"body": {"command_delay_ms": 20}``
    the servo command delay the table was gated with. The Stage-1 skill networks were trained
    with training's 15–30 ms delay and need it (tr:gate 2026-09-28: on the simulator's XML
    servos with 0 ms the hop-back falls 30/30 and the shake stays under its gate; with 20 ms
    both pass, 0 falls in 2,463 trials). The launcher gives it to the body server
    (``duckmind.bodyhost``'s ``DUCKMIND_BODY_COMMAND_DELAY_MS``) unless that is set already.

``"guard": {"global_per_hour": 60, "rules": {name: {"per_hour", "min_gap_s", "postures",
"max_walk_mps", "settle_s", "rise_settle_s", "hold", "continues"}}}``
    duck-body's start rule and wear budget for this table (``duckmind.skilltrain.guard``): the
    gate's verdict (a skill allowed only in the postures that passed, from a walk only up to the
    fastest speed that passed), the settle times, the held crouches, and the hourly caps set
    from its strain numbers. Missing entries fall back to :data:`guard.DEFAULT_RULES`.

``"files": {role: {"path", "sha256"}}``
    (s1b:fix) every network the table was gated with, pinned: the two skill networks **and**
    the walk network ``skill_abort`` runs and every skill hands back to (``walk``). Because the
    owner accepts OTA updates, a changed velstand would silently change the abort and the
    hand-back the gates measured: :func:`check_pins` finds any file whose sha256 differs, and
    :func:`verified_section` (what the launcher calls) refuses such a table.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from duckmind.skilltrain import guard as G
from duckmind.skilltrain.table import SKILLS_ENV, load, toml_section

DELAY_ENV = "DUCKMIND_BODY_COMMAND_DELAY_MS"  # duckmind.bodyhost.COMMAND_DELAY_ENV


def table_doc(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The installed table (``$DUCKMIND_ROBOTD_SKILLS``) as a dict; {} when none."""
    env = os.environ if environ is None else environ
    raw = (env.get(SKILLS_ENV) or "").strip()
    if not raw:
        return {}
    doc = json.loads(Path(raw).read_text(encoding="utf-8"))
    return doc if isinstance(doc, dict) else {}


def command_delay_ms(doc: Mapping[str, Any]) -> float | None:
    """The servo command delay the table asks for (None: it asks for none)."""
    body = doc.get("body")
    if not isinstance(body, Mapping) or body.get("command_delay_ms") is None:
        return None
    ms = float(body["command_delay_ms"])
    if not (0.0 <= ms <= 100.0):
        raise ValueError(f"body.command_delay_ms {ms}: 0 to 100 ms")
    return ms


def body_env(
    base: Mapping[str, str] | None = None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """``base`` plus the table's servo command delay for the body server (an explicit
    ``$DUCKMIND_BODY_COMMAND_DELAY_MS`` wins; no table or no delay: ``base`` unchanged)."""
    env = dict(os.environ if base is None else base)
    source = os.environ if environ is None else environ
    if (source.get(DELAY_ENV) or "").strip():
        env[DELAY_ENV] = str(source[DELAY_ENV]).strip()
        return env
    ms = command_delay_ms(table_doc(source))
    if ms is not None:
        env[DELAY_ENV] = f"{ms:g}"
    return env


def none_slots(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """The robotd skill slots the installed table switches off (``"policy": {"none_slots": [...]}``:
    ``roulade``, to make room under robotd's 30 entries — the slot's own line in ``[policy]`` outranks
    a ``[[policy.skill]]`` entry that names it, so the launcher has to write it)."""
    block = table_doc(environ).get("policy")
    if not isinstance(block, Mapping):
        return ()
    return tuple(str(s) for s in block.get("none_slots") or () if str(s) in ("roulade",))


def guard_rules(
    doc: Mapping[str, Any], base: Mapping[str, G.Rule] | None = None
) -> dict[str, G.Rule]:
    """The guard's rules for this table: ``base`` (default :data:`guard.DEFAULT_RULES`) with the
    table's per-skill ``per_hour`` / ``min_gap_s`` / ``postures`` applied."""
    rules = dict(G.DEFAULT_RULES if base is None else base)
    block = doc.get("guard")
    if not isinstance(block, Mapping):
        return rules
    for name, spec in (block.get("rules") or {}).items():
        if not isinstance(spec, Mapping):
            continue
        old = rules.get(name)
        if old is None:
            continue
        per_hour = spec.get("per_hour", old.per_hour)
        postures: Iterable[str] = spec.get("postures") or old.postures
        top = spec.get("max_walk_mps", old.max_walk_mps)
        continues: Iterable[str] = spec.get("continues", old.continues)
        abort_tilt = spec.get("abort_tilt_deg", old.abort_tilt_deg)
        rules[name] = replace(
            old,
            per_hour=None if per_hour is None else int(per_hour),
            min_gap_s=float(spec.get("min_gap_s", old.min_gap_s)),
            postures=frozenset(str(p) for p in postures),
            note=str(spec.get("note", old.note)),
            max_walk_mps=None if top is None else float(top),
            settle_s=float(spec.get("settle_s", old.settle_s)),
            rise_settle_s=float(spec.get("rise_settle_s", old.rise_settle_s)),
            hold=bool(spec.get("hold", old.hold)),
            continues=frozenset(str(f) for f in continues),
            abort_tilt_deg=None if abort_tilt is None else float(abort_tilt),
            free_continue=bool(spec.get("free_continue", old.free_continue)),
            counted=bool(spec.get("counted", old.counted)),
        )
    return rules


def sha256_of(path: str | os.PathLike[str]) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pins(doc: Mapping[str, Any]) -> list[str]:
    """What differs from the files the table was gated with: each ``files`` entry whose sha256
    no longer matches, and each skill path no ``files`` entry pins (a table from before the pins
    reports its walk network as unpinned)."""
    problems: list[str] = []
    files = doc.get("files") or {}
    pinned: dict[str, str] = {}
    for role, spec in files.items():
        if not isinstance(spec, Mapping) or not spec.get("path"):
            continue
        path, want = str(spec["path"]), spec.get("sha256")
        pinned[str(Path(path).resolve())] = str(role)
        got = sha256_of(path)
        if got is None:
            problems.append(f"{role}: {path} is missing")
        elif want and got != want:
            problems.append(
                f"{role}: {path} changed (sha256 {got[:12]}… ≠ gated {str(want)[:12]}…)"
            )
        elif not want:
            problems.append(f"{role}: {path} has no sha256 pin")
    for raw in doc.get("skills") or []:
        path = raw.get("path")
        if path and str(path).lower() != "none" and str(Path(path).resolve()) not in pinned:
            problems.append(f"skill {raw.get('name')!r}: {path} is not pinned in the table's files")
    return problems


def verified_section(environ: Mapping[str, str] | None = None) -> str:
    """The ``[[policy.skill]]`` section of the installed table, **only if** every network it
    runs is the one it was gated with (:func:`check_pins`); '' when no table is installed.
    Raises ValueError otherwise — the launcher refuses to start (a simulator running networks
    its gates never saw is worse than one that says so)."""
    env = os.environ if environ is None else environ
    raw = (env.get(SKILLS_ENV) or "").strip()
    if not raw:
        return ""
    doc = table_doc(env)
    problems = check_pins(doc)
    if problems:
        raise ValueError(
            "the skill table's networks are not the gated ones: " + "; ".join(problems)
        )
    return toml_section(load(raw))


def global_per_hour(doc: Mapping[str, Any]) -> int:
    block = doc.get("guard")
    if isinstance(block, Mapping) and block.get("global_per_hour") is not None:
        return int(block["global_per_hour"])
    return G.GLOBAL_PER_HOUR


def skill_guard(available: Iterable[str], environ: Mapping[str, str] | None = None) -> G.SkillGuard:
    """A :class:`guard.SkillGuard` for the skills robotd loaded, with the installed table's
    rules (the provisional defaults when the table carries none)."""
    doc = table_doc(environ)
    return G.SkillGuard(available, guard_rules(doc), global_per_hour=global_per_hour(doc))


def pick_loaded(skill: str | None, loaded: Iterable[str]) -> str | None:
    """The picker's choice if this robotd loaded it (a table without shakes never logs a
    refusal for every shake bout)."""
    return skill if skill is not None and skill in set(loaded) else None


__all__ = [
    "DELAY_ENV",
    "body_env",
    "check_pins",
    "command_delay_ms",
    "global_per_hour",
    "guard_rules",
    "none_slots",
    "pick_loaded",
    "skill_guard",
    "sha256_of",
    "table_doc",
    "verified_section",
]
