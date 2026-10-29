"""Per-episode records → the numbers in a sweep result, and back again from the JSONL file.

Every scenario keeps one raw row per simulated duck (``<result>.trials.jsonl``, first line a
header with the schedule and rules). The runner builds the result's summaries with the
functions here, and :func:`recompute` rebuilds them from the JSONL alone, so anyone can check
a number or count it another way. Pure Python: no torch, no mjlab.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from duckmind.expression_sweep import falls, looks
from duckmind.expression_sweep.stats import describe, difference, percentile, proportion

POSTURES = ("sit", "stand")


def _judged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Episodes in the posture when the clip began and never reset mid-episode.

    With none, all of them: a clip is never called safe on no evidence.
    """
    kept = [row for row in rows if not row.get("settle_bad") and not row.get("reset")]
    return kept or rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variant_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One variant in one posture: the v2 statistics, plus the new ones when recorded."""
    judged = _judged(rows)

    def pick(name: str) -> list[Any]:
        return [row[name] for row in judged]

    stats: dict[str, Any] = {
        "episodes": len(rows),
        "settle_failures": sum(1 for row in rows if row.get("settle_bad")),
        "falls": int(sum(bool(value) for value in pick("fell"))),
        "tilt_rise_deg_p95": round(percentile(pick("tilt_rise"), 0.95), 2),
        "tilt_deg_p95": round(percentile(pick("tilt"), 0.95), 2),
        "torque_ratio_max": round(max(pick("torque_ratio")), 3),
        "strain_frac_p95": round(percentile(pick("strain_frac"), 0.95), 3),
        "self_contacts": int(sum(1 for value in pick("contacts") if value > 0)),
        "track_ratio": round(_mean(pick("track_ratio")), 3),
        "rms_error": round(_mean(pick("rms_error")), 4),
        "drift_m_p95": round(percentile(pick("drift"), 0.95), 4),
    }
    if all("fell_robotd" in row for row in judged):
        stats["falls_robotd"] = int(sum(bool(row["fell_robotd"]) for row in judged))
    if any("reset" in row for row in rows):
        stats["resets"] = sum(1 for row in rows if row.get("reset"))
    if any(row.get("cam_pitch") for row in judged):
        stats["rest_pitch_deg"] = round(_mean([row["rest_pitch"] for row in judged]), 2)
        captures = []
        for k in range(max(len(row.get("cam_pitch") or ()) for row in judged)):
            pitches = _capture(judged, "cam_pitch", k)
            if not pitches:
                continue
            captures.append(
                {
                    "pitch_deg": round(_mean(pitches), 2),
                    "pitch_deg_p5": round(percentile(pitches, 0.05), 2),
                    "pitch_deg_p95": round(percentile(pitches, 0.95), 2),
                    "height_m": round(_mean(_capture(judged, "cam_height", k)), 4),
                    "still": round(_mean(_capture(judged, "cam_still", k)), 3),
                }
            )
        stats["captures"] = captures
    return stats


def _capture(rows: list[dict[str, Any]], name: str, k: int) -> list[float]:
    values = []
    for row in rows:
        series = row.get(name) or ()
        if k < len(series) and series[k] is not None and math.isfinite(float(series[k])):
            values.append(float(series[k]))
    return values


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{variant key: stats} for rows of one posture, in first-seen order."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(row["variant"], []).append(row)
    return {key: variant_stats(group) for key, group in by_key.items()}


def aggregate(assigned: list[Any], metrics: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Column-wise metrics for ``assigned[i]`` (a :class:`~.clips.Variant`) → per-variant stats."""
    rows = [
        {"variant": variant.key} | {name: values[i] for name, values in metrics.items()}
        for i, variant in enumerate(assigned)
    ]
    return aggregate_rows(rows)


def by_posture(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The result's ``{variant: {posture: stats}}`` shape from head or look rows."""
    out: dict[str, dict[str, Any]] = {}
    for posture in POSTURES:
        for key, stats in aggregate_rows(row for row in rows if row["posture"] == posture).items():
            out.setdefault(key, {})[posture] = stats
    return out


def _look_block(rows: list[dict[str, Any]], control_drift_p95: float | None) -> dict[str, Any]:
    judged = [row for row in rows if not row.get("settle_bad") and not row.get("reset")]
    drift = [row["drift"] for row in judged]
    captures = []
    for k in range(looks.MAX_CAPTURES):
        pitches = _capture(judged, "cam_pitch", k)
        if not pitches:
            continue
        stills = _capture(judged, "cam_still", k)
        captures.append(
            {
                "pitch_deg": describe(pitches, digits=2),
                "height_m": describe(_capture(judged, "cam_height", k), digits=4),
                "still": proportion(int(sum(stills)), len(stills)),
            }
        )
    block: dict[str, Any] = {
        "episodes": len(rows),
        "settle_failures": sum(1 for row in rows if row.get("settle_bad")),
        "resets": sum(1 for row in rows if row.get("reset")),
        "falls_robotd": proportion(sum(1 for row in judged if row["fell_robotd"]), len(judged)),
        "falls_45": proportion(sum(1 for row in judged if row["fell"]), len(judged)),
        "tilt_deg": describe([row["tilt"] for row in judged], digits=2),
        "tilt_rise_deg": describe([row["tilt_rise"] for row in judged], digits=2),
        "drift_m": describe(drift, digits=4),
        "rest_pitch_deg": describe([row["rest_pitch"] for row in judged], digits=2),
        "rest_height_m": describe([row["rest_height"] for row in judged], digits=4),
        "captures": captures,
    }
    if control_drift_p95 is not None and drift:
        over = percentile(drift, 0.95) - control_drift_p95
        block["drift_p95_over_control_m"] = round(over, 4)
        block["gate"] = {
            "passes": block["falls_robotd"]["count"] == 0
            and block["falls_45"]["count"] == 0
            and over <= looks.DRIFT_MARGIN_M,
            "rule": (
                "design amplitude, every tempo and side: 0 falls by robotd's rule and by 45°, "
                f"drift p95 ≤ the HOME-hold control's + {looks.DRIFT_MARGIN_M} m"
            ),
        }
    return block


def summarize_looks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per look and posture: the design pose (amplitude 1.0) against the HOME-hold control."""
    out: dict[str, Any] = {}
    for posture in POSTURES:
        mine = [row for row in rows if row["posture"] == posture]
        control = [row for row in mine if row["clip"] == looks.CONTROL]
        if not mine:
            continue
        control_block = _look_block(control, None) if control else None
        control_p95 = (
            percentile(
                [r["drift"] for r in control if not r.get("settle_bad") and not r.get("reset")],
                0.95,
            )
            if control
            else None
        )
        if control_block is not None:
            out.setdefault(looks.CONTROL, {})[posture] = control_block
        for clip in looks.LOOKS:
            clip_rows = [row for row in mine if row["clip"] == clip]
            if not clip_rows:
                continue
            design = [row for row in clip_rows if row["amplitude"] == 1.0]
            block = _look_block(design, control_p95)
            judged = [r for r in clip_rows if not r.get("settle_bad") and not r.get("reset")]
            block["whole_grid"] = {
                "episodes": len(clip_rows),
                "falls_robotd": proportion(sum(1 for r in judged if r["fell_robotd"]), len(judged)),
                "falls_45": proportion(sum(1 for r in judged if r["fell"]), len(judged)),
            }
            out.setdefault(clip, {})[posture] = block
    return out


def summarize_head_walk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per head pose: speed and yaw of the walk, falls, and where the camera pointed."""
    out: dict[str, Any] = {}
    labels = sorted({row["head"] for row in rows}, key=lambda label: label != "level")
    for label in labels:
        group = [row for row in rows if row["head"] == label and not row.get("reset")]
        walked = [row for row in group if not row["fell_robotd"]]
        first = group[0] if group else {}
        out[label] = {
            "commanded": {
                key: first.get(key) for key in ("vx", "vyaw", "seconds", "head_offsets_rad")
            },
            "episodes": len([row for row in rows if row["head"] == label]),
            "resets": sum(1 for row in rows if row["head"] == label and row.get("reset")),
            "falls_robotd": proportion(sum(1 for row in group if row["fell_robotd"]), len(group)),
            "falls_45": proportion(sum(1 for row in group if row["fell"]), len(group)),
            "speed_mps": describe([row["speed_mps"] for row in walked]),
            "steady_speed_mps": describe([row["steady_speed_mps"] for row in walked]),
            "yaw_rate_dps": describe([row["yaw_rate_dps"] for row in walked], digits=2),
            "steady_yaw_rate_dps": describe(
                [row["steady_yaw_rate_dps"] for row in walked], digits=2
            ),
            "distance_m": describe([row["distance_m"] for row in walked]),
            "overshoot_m": describe([row["overshoot_m"] for row in walked]),
            "tilt_deg": describe([row["tilt"] for row in group], digits=2),
            "cam_pitch_deg": describe([row["cam_pitch_deg"] for row in walked], digits=2),
            "cam_height_m": describe([row["cam_height_m"] for row in walked]),
        }
    if "level" in out:
        level = [
            row
            for row in rows
            if row["head"] == "level" and not row.get("reset") and not row["fell_robotd"]
        ]
        compare = {}
        for label in out:
            if label == "level":
                continue
            other = [
                row
                for row in rows
                if row["head"] == label and not row.get("reset") and not row["fell_robotd"]
            ]
            compare[label] = {
                name: difference([row[name] for row in other], [row[name] for row in level])
                for name in ("speed_mps", "steady_speed_mps", "yaw_rate_dps", "steady_yaw_rate_dps")
            }
        out["compare_to_level"] = compare
    return out


def summarize(header: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every summary a v3 result carries, from its raw rows."""
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    out: dict[str, Any] = {}
    if "posture" in by_scenario:
        schedule = falls.PostureSchedule.from_dict(header["schedule"])
        tasks = sorted({row["task"] for row in by_scenario["posture"]})
        out["posture"] = {
            task: falls.summarize_posture(
                [row for row in by_scenario["posture"] if row["task"] == task], schedule
            )
            for task in tasks
        }
    if "look" in by_scenario:
        out["looks"] = by_posture(by_scenario["look"])
        out["looks_summary"] = summarize_looks(by_scenario["look"])
    if "head" in by_scenario:
        out["variants"] = by_posture(by_scenario["head"])
    if "headwalk" in by_scenario:
        out["head_walk"] = summarize_head_walk(by_scenario["headwalk"])
    return out


def write_rows(path: Path, rows: Iterable[dict[str, Any]], *, mode: str = "a") -> int:
    count = 0
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_trials(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scenario") == "header":
                header = row
            else:
                rows.append(row)
    return header, rows


def recompute(path: Path) -> dict[str, Any]:
    """The summaries of a v3 result, rebuilt from its ``.trials.jsonl`` alone."""
    header, rows = read_trials(path)
    return summarize(header, rows)
