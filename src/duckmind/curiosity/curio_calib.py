"""curio_calib: the curiosity numbers from measurement records (curiosity spec §5.2 "可達地圖",
§13 ``curiosity-outcomes-v1``; phase 3 re-measurement under the robotd path).

The live measurement driver is in the measurement records, not in the repo (the phase-3
curiosity ``measure.py``: official ``robotd --sim`` in my own world server, truth used only to place things
and to score what happened); its JSONL records come here. :func:`summarise` turns them into the
calibration summary — the evidence file ``docs/validation/evidence/phase3/curiosity_calib.json``
— with every rate's n and Wilson 95 % interval:

* ``beak`` — where ground_pick's beak tip lands (trunk frame at the start), when, and how the
  trunk rocks meanwhile;
* ``steps`` — how far each primitive moves the trunk from standing (creep, trim, bump, leg, back,
  reverse arcs, head-down creep), its yaw, falls;
* ``kick`` — the ball's roll per spot and foot, the window where it rolls, the kick direction per
  foot and the roll distances (the kick cone's inputs);
* ``peck``, ``bump`` — contact and displacement per thing and standoff;
* ``grab`` — the jaw's lift window (provisional, owner A4), the mouth angle held / empty, the relay;
* ``looks`` — the look poses held standing (falls, drift).

:func:`outcome_rows` feeds :class:`~duckmind.curiosity.outcomes.OutcomeTable` from Act Runner
``body.curio_step`` records.

Pure: standard library only (records in, JSON out).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from duckmind.curiosity.outcomes import OutcomeTable, Row, quantiles, rate

ROLLED_MM = 300
LIFT_MM = 30
SCHEMA = "curiosity-calib-v1"


def load(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]
    return records


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _mm(value: float | None) -> int | None:
    return None if value is None else int(round(value * 1000.0))


def _q(values: Sequence[float]) -> list[int] | None:
    return quantiles(values, (0.1, 0.5, 0.9))


def _med(values: Sequence[float]) -> float | None:
    data = sorted(values)
    if not data:
        return None
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2


def beak(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [r for r in records if r.get("task") == "beak" and r.get("lowest")]
    if not rows:
        return None
    tips = [r["lowest"]["tip"] for r in rows]
    t0 = [r["rows"][0]["t"] for r in rows]
    return {
        "n": len(rows),
        "tip_lowest_ahead_mm": _q([t[0] * 1000 for t in tips]),
        "tip_lowest_left_mm": _q([t[1] * 1000 for t in tips]),
        "tip_lowest_height_mm": _q([t[2] for t in tips]),
        "tip_at_start_ahead_mm": _q([r["tip_start"][0] * 1000 for r in rows]),
        "tip_at_start_height_mm": _q([r["tip_start"][2] for r in rows]),
        "bottom_ms": _q([(r["lowest"]["t"] - s) * 1000 for r, s in zip(rows, t0, strict=True)]),
        "trunk_at_bottom_ahead_mm": _q([r["lowest"]["ahead"] * 1000 for r in rows]),
        "trunk_at_bottom_dyaw_mrad": _q([r["lowest"]["dyaw"] * 1000 for r in rows]),
        "skill_ms": _q([(r["skill"]["back_s"] or 0) * 1000 for r in rows]),
        "max_tilt_deg": max(r["max_tilt"] for r in rows),
        "note": "the beak tip (official mouth_tip site) in the trunk frame at the start, standing: "
        "the lowest point is where a peck reaches",
    }


def steps(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("task") in ("creep", "moves"):
            groups[str(r["label"])].append(r)
    out = {}
    for label, rows in sorted(groups.items()):
        falls = sum(1 for r in rows if r.get("fell"))
        out[label] = {
            "vx": rows[0]["vx"],
            "wz": rows[0].get("wz", 0.0),
            "secs": rows[0]["secs"],
            "head": rows[0].get("head", "home"),
            "n": len(rows),
            "ahead_mm_p10_p50_p90": _q([r["ahead_m"] * 1000 for r in rows]),
            "left_mm_p10_p50_p90": _q([r["left_m"] * 1000 for r in rows]),
            "dyaw_mrad_p10_p50_p90": _q([r["dyaw_rad"] * 1000 for r in rows]),
            "max_tilt_deg": max(r["max_tilt"] for r in rows if r.get("max_tilt") is not None),
            "falls": rate(falls, len(rows)),
        }
    return out


def kick(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    rows = [r for r in records if r.get("task") == "kick"]
    if not rows:
        return None
    grid: dict[str, dict[str, dict[str, Any]]] = {"left": {}, "right": {}}
    for r in rows:
        key = f"{_mm(r['ahead'])},{_mm(r['side'])}"
        cell = grid[r["foot"]].setdefault(key, {"n": 0, "placed": 0, "rolled": 0, "moved_mm": []})
        cell["n"] += 1
        if r["placed"].get("ok"):
            cell["placed"] += 1
            cell["moved_mm"].append(int(round(r["moved_mm"] or 0)))
            cell["rolled"] += int((r["moved_mm"] or 0) >= ROLLED_MM)
    per_foot = {}
    for foot in ("left", "right"):
        ok = [r for r in rows if r["foot"] == foot and r["placed"].get("ok")]
        rolled = [r for r in ok if (r["moved_mm"] or 0) >= ROLLED_MM]
        per_foot[foot] = {
            "tried": len(ok),
            "rolled": rate(len(rolled), len(ok)),
            "dir_mrad_p10_p50_p90": _q(
                [r["dir_rad"] * 1000 for r in rolled if r.get("dir_rad") is not None]
            ),
            "roll_mm_p10_p50_p90": _q([r["moved_mm"] for r in rolled]),
        }
    falls = sum(1 for r in rows if r.get("fell"))
    return {
        "n": len(rows),
        "grid": grid,
        "per_foot": per_foot,
        "falls": rate(falls, len(rows)),
        "max_tilt_deg": max(r["max_tilt"] for r in rows if r.get("max_tilt") is not None),
        "note": "ball_felt (25 g, 70 mm, rolling friction 0.004) on default_home's wood floor; "
        "'ahead' and 'side' are the ball centre in the trunk frame, side toward the kicking foot; "
        "a spot the world refused (inside the duck's feet) counts as not placed",
    }


def window_of(kick_summary: Mapping[str, Any], *, min_share: float = 1.0) -> dict[str, Any]:
    """The largest rectangle of grid spots where every placed kick rolled, for both feet."""
    spots: dict[tuple[int, int], bool] = {}
    for foot in ("left", "right"):
        for key, cell in kick_summary["grid"][foot].items():
            a, s = (int(v) for v in key.split(","))
            good = cell["placed"] > 0 and cell["rolled"] >= min_share * cell["placed"]
            spots[(a, s)] = spots.get((a, s), True) and good
    aheads = sorted({a for a, _ in spots})
    sides = sorted({s for _, s in spots})
    best: tuple[int, int, int, int, int] | None = None
    for i0 in range(len(aheads)):
        for i1 in range(i0, len(aheads)):
            for j0 in range(len(sides)):
                for j1 in range(j0, len(sides)):
                    cells = [(a, s) for a in aheads[i0 : i1 + 1] for s in sides[j0 : j1 + 1]]
                    if all(spots.get(c, False) for c in cells):
                        area = len(cells)
                        if best is None or area > best[0]:
                            best = (area, aheads[i0], aheads[i1], sides[j0], sides[j1])
    if best is None:
        return {"ahead_mm": None, "side_mm": None}
    return {"ahead_mm": [best[1], best[2]], "side_mm": [best[3], best[4]], "spots": best[0]}


def peck(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("task") == "peck":
            groups[f"{r['thing']}@{_mm(r['ahead'])}"].append(r)
    out = {}
    for key, rows in sorted(groups.items()):
        placed = [r for r in rows if r["placed"].get("ok")]
        touched = [r for r in placed if any(b in ("jaw_soft", "mandible") for b in r["touched_by"])]
        out[key] = {
            "n": len(rows),
            "placed": len(placed),
            "beak_touched": rate(len(touched), len(placed)) if placed else None,
            "moved_mm_p50": _med([r["moved_mm"] for r in placed if r["moved_mm"] is not None]),
            "first_touch_ms": _q(
                [r["first_touch_t"] * 1000 for r in touched if r.get("first_touch_t") is not None]
            ),
        }
    return out


def bump(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("task") == "bump":
            groups[f"{r['thing']}@100"].append(r)
        elif r.get("task") == "bump2":
            groups[f"{r['thing']}@{_mm(r['gap'])}"].append(r)
    out = {}
    for key, rows in sorted(groups.items()):
        placed = [r for r in rows if r["placed"].get("ok")]
        moved = [r for r in placed if (r["moved_mm"] or 0) >= 15]
        touched = [r for r in placed if r.get("touched_by")]
        out[key] = {
            "n": len(rows),
            "placed": len(placed),
            "touched": rate(len(touched), len(placed)) if placed else None,
            "moved_15mm": rate(len(moved), len(placed)) if placed else None,
            "moved_mm_p10_p50_p90": _q(
                [r["moved_mm"] for r in placed if r["moved_mm"] is not None]
            ),
            "duck_ahead_mm_p50": _med([r["duck_ahead_m"] * 1000 for r in placed]),
            "falls": rate(sum(1 for r in rows if r.get("fell")), len(rows)),
        }
    return out


def grab(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("task") in ("grab", "grab2"):
            relay = r.get("relay_s", 1.0)
            groups[f"{_mm(r['ahead'])},{_mm(r['left'])}@{int(relay * 1000)}ms"].append(r)
    cells = {}
    held_deg: list[float] = []
    empty_deg: list[float] = []
    for key, rows in sorted(groups.items()):
        placed = [r for r in rows if r["placed"].get("ok")]
        lifted = [r for r in placed if r.get("lifted")]
        for r in placed:
            (held_deg if r.get("lifted") else empty_deg).append(r["mouth_after_close_deg"])
        cells[key] = {
            "n": len(rows),
            "placed": len(placed),
            "lifted": rate(len(lifted), len(placed)) if placed else None,
            "held_until_ms": _q(
                [r["lift_last_t"] * 1000 for r in lifted if r.get("lift_last_t") is not None]
            ),
        }
    return {
        "cells": cells,
        "mouth_deg_held_p10_p50_p90": _q(held_deg),
        "mouth_deg_empty_p10_p50_p90": _q(empty_deg),
        "provisional": True,
        "note": "owner decision A4: the jaw's pivot is a guess (world/jaw.py); fin_fish (30 g, 5 cm "
        "fin) placed across the beak; 'lifted' = the fin rose > 30 mm; held_until counts from "
        "the pick's start (the relay leg starts ~3.9 s in and the beak opens at its end)",
    }


def looks(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("task") == "looks":
            groups[str(r["label"])].append(r)
    return {
        label: {
            "n": len(rows),
            "falls": rate(sum(1 for r in rows if r.get("fell")), len(rows)),
            "drift_mm_p50": _med([r["drift_mm"] for r in rows]),
            "max_tilt_deg": max(r["max_tilt"] for r in rows if r.get("max_tilt") is not None),
        }
        for label, rows in sorted(groups.items())
    }


def summarise(
    records: Sequence[Mapping[str, Any]], *, sources: Mapping[str, str]
) -> dict[str, Any]:
    kicks = kick(records)
    return {
        "schema": SCHEMA,
        "physics": "live: official robotd --sim (lockstep, allcollisions) + duckmind world server, "
        "default_home wood floor, provisional jaw (owner A4)",
        "sources": dict(sources),
        "beak": beak(records),
        "steps": steps(records),
        "kick": kicks,
        "kick_window": None if kicks is None else window_of(kicks),
        "peck": peck(records),
        "bump": bump(records),
        "grab": grab(records),
        "looks": looks(records),
    }


MOVED_WORDS = ("moved", "rolled", "tipped", "squirted")
TRUTH_WORKED = {  # what truth calls "it worked" per verb (evaluation only: never Duck Core's)
    "kick": lambda r: (r.get("truth_moved_mm") or 0) >= ROLLED_MM,
    "dribble": lambda r: (r.get("truth_moved_mm") or 0) >= 30,
    "peck": lambda r: (r.get("truth_moved_mm") or 0) >= 15,
    "nibble": lambda r: (r.get("truth_moved_mm") or 0) >= 15,
    "bump": lambda r: (r.get("truth_moved_mm") or 0) >= 15,
    "grab": lambda r: (r.get("lifted_before_carry_mm") or 0) >= 45,
    "look": lambda r: bool((r.get("step") or {}).get("thing_after", {}).get("seen")),
}


def _report_says(verb: str, step: Mapping[str, Any]) -> bool | None:
    """What the duck's own report says about the same question (None: it cannot tell)."""
    outcome = step.get("outcome")
    if verb == "grab":
        return outcome == "held"
    if verb == "look":
        return outcome == "seen"
    if verb == "kick":
        if outcome == "moved" and step.get("lost") and step.get("moved_mm") is None:
            return None  # gone from its spot, not found: how far is unknown
        return outcome == "rolled"
    return outcome in MOVED_WORDS or outcome == "in_beak"


def runner(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Live Act Runner spot checks (``runner_live.py`` records): per verb, how often the step
    worked by truth, how often the duck's own report said the same as truth (and how often it
    could not tell), falls, the body loop guard's refusals, rule violations, the step's time."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r["verb"])].append(r)
    out = {}
    for verb, rows in sorted(groups.items()):
        done = [r for r in rows if r.get("step")]
        worked = TRUTH_WORKED.get(verb, lambda r: False)
        judged = [(r, _report_says(verb, r["step"])) for r in done]
        known = [(r, said) for r, said in judged if said is not None]
        agree = [r for r, said in known if said == worked(r)]
        falls = [r for r in done if r["step"].get("fell")]
        cell: dict[str, Any] = {
            "n": len(rows),
            "reported": len(done),
            "worked_by_truth": rate(sum(1 for r in done if worked(r)), len(done)),
            "report_agrees_with_truth": rate(len(agree), len(known)) if known else None,
            "report_cannot_tell": len(judged) - len(known),
            "outcomes": dict(sorted(Counter(str(r["step"].get("outcome")) for r in done).items())),
            "ended_by": dict(sorted(Counter(str(r["step"].get("ended_by")) for r in done).items())),
            "truth_moved_mm_p10_p50_p90": _q(
                [r["truth_moved_mm"] for r in done if r.get("truth_moved_mm") is not None]
            ),
            "falls": rate(len(falls), len(done)),
            "guard_refusals_per_step": _q([r.get("refused_n", 0) for r in rows]),
            "violations": sum(len(r.get("violations") or []) for r in rows),
            "report_problems": sum(
                1 for r in done if [p for p in r.get("problems", []) if "for Duck Core" not in p]
            ),
            "step_ms_p10_p50_p90": _q([r["step"]["ms"] for r in done]),
            "looks_p50": _med([r["step"]["looks"] for r in done]),
            "head_fallbacks": sum(r["step"].get("head_fallback", 0) for r in done),
            "blind_steps": sum(1 for r in done if r["step"].get("blind_step")),
        }
        if verb == "grab":
            carried = [r for r in rows if r.get("carry")]
            cell["carry_held_by_report"] = (
                rate(
                    sum(
                        1
                        for r in carried
                        if (r["carry"].get("step") or {}).get("outcome") == "held"
                    ),
                    len(carried),
                )
                if carried
                else None
            )
            cell["carry_truth_moved_mm"] = _q(
                [
                    r["carry_truth_moved_mm"]
                    for r in carried
                    if r.get("carry_truth_moved_mm") is not None
                ]
            )
        out[verb] = cell
    return out


def outcome_rows(
    steps_: Iterable[Mapping[str, Any]], *, cls_of: Mapping[str, str] | None = None
) -> list[Row]:
    """Act Runner ``body.curio_step`` events → outcome-table rows (the tier is in the event)."""
    rows = []
    for step in steps_:
        if step.get("kind", "body.curio_step") != "body.curio_step":
            continue
        cls_name = str(step.get("cls") or (cls_of or {}).get(str(step.get("thing")), "unknown"))
        rows.append(Row.from_step(step, cls_name=cls_name, tier=str(step.get("tier", "normal"))))
    return rows


def outcome_table(
    steps_: Iterable[Mapping[str, Any]], *, physics: str, perception: str
) -> dict[str, Any]:
    table = OutcomeTable(physics=physics, perception=perception)
    table.extend(outcome_rows(steps_))
    return table.to_json()


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("records", nargs="+", help="measure.py (or runner_live.py) JSONL files")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--runner",
        action="store_true",
        help="the records are live Act Runner steps: write the spot-check summary "
        "and the outcome table instead",
    )
    args = parser.parse_args(argv)
    records = load(args.records)
    sources = {str(Path(p).name): sha256(p) for p in args.records}
    if args.runner:
        steps_ = [r["step"] for r in records if r.get("step")]
        steps_ += [r["carry"]["step"] for r in records if (r.get("carry") or {}).get("step")]
        summary = {
            "schema": "curiosity-runner-live-v1",
            "physics": "live: official robotd --sim (lockstep, allcollisions) + duckmind world "
            "server, default_home wood floor, provisional jaw (owner A4)",
            "perception": "oracle_noise (the M2 camera stand-in: truth + field of view + noise); "
            "NOT the camera gate — duck-eye's things task replaces it",
            "sources": sources,
            "runner": runner(records),
            "outcome_table": outcome_table(
                steps_, physics="live robotd --sim", perception="oracle_noise"
            ),
        }
    else:
        summary = summarise(records, sources=sources)
    Path(args.out).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
