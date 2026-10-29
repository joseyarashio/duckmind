"""Stage-2 abilities (s2:build): which of a walk network's new abilities passed their gates — the
``abilities.json`` the neural demo's wiring reads beside the ONNX (``duckmind.neuro.walknet`` in the
neuro worktree; ``wiring/README.md`` of the Stage-2 data folder).

Sources: the A100 battery (:mod:`s2_battery`: ``gates`` per servo delay, the training's BAM servo
model, n = 512 DR ducks per condition — the only source with the n to resolve the rate gates) and
the Mac harness summaries (:mod:`s2_mac` ``summarize``: ``verdicts`` per servo model / delay / arm,
n = 30 DR ducks per cell, both servo models). The rules (pre-registered here, never tuned to a
result):

* a gate that FAILs in any source is not passed;
* a rate gate (falls, near-falls, stepping, walk-offs, righting within 8 s) passes when some source
  resolves it as PASS (in practice the A100 battery) and none FAILs; UNRESOLVED Mac cells are
  allowed and listed;
* every other gate (degrees, mm, lags, ratios) must PASS on BOTH servo models (``bam`` and
  ``mjcf``) — a model that was not measured means "not passed";
* ``usable`` = every walking regression gate of the stage passed (it walks, stops, turns, rights
  itself and sits/stands at least as well as v5). An unusable network grants nothing;
* a pose program needs the pose safety gates (falls and stepping while posed, the release rate) and
  its own visibility gates; the neck and the steady gaze have their own.

``python -m duckmind.skilltrain.s2_abilities --walk final/velstand_duck.onnx --stage gaze
--battery battery-full.json --mac mac/summary.json --arm r4 --out final/abilities.json``
Pure standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from duckmind.skilltrain import s2_contract as C

SCHEMA = 1
SERVOS = ("bam", "mjcf")
RATE_KINDS = ("rate_max", "rate_min", "max_vs_v5")
REGRESSION = ("walk_falls", "stopgo_near_falls", "push_righting_8s", "righting_p50_extra_s",
              "sitstand_falls", "walk_speed_ratio", "turn_gain_ratio", "stop_s_p50",
              "head_motion_falls", "stand_stepped", "walkoffs", "stand_leg_rms_ratio",
              "cam_perp_regress_ratio")  # fmt: skip
GROUPS: dict[str, tuple[str, ...]] = {
    "usable": REGRESSION,
    "pose_safe": ("pose_falls", "pose_stepped", "release_rate_dps"),
    "lean": ("lean_roll_deg", "lean_pitch_deg", "weight_shift_ratio"),
    "sway": ("sway_amp_deg", "sway_lag_ms"),
    "chest_out": ("chest_out_deg",),
    "crouch": ("crouch_mm",),
    "neck_standing": ("neck_pullback_gain", "yaw_micro_gain"),
    "gaze_stable": ("cam_perp_ratio", "head_yaw_stride_sd_ratio", "head_yaw_delivered"),
}
POSE_PROGRAMS = ("lean", "sway", "chest_out", "crouch")
GROUP_ZH = {
    "usable": "走路、停、轉彎、跌倒爬起、坐下站起都不比 v5 差",
    "pose_safe": "站著擺姿勢時不跌倒、不亂踏步、放掉時不會猛晃",
    "lean": "站著往左右、前後傾",
    "sway": "站著身體左右晃",
    "chest_out": "挺胸站高",
    "crouch": "蹲低",
    "neck_standing": "站著時脖子也會跟著小動作",
    "gaze_stable": "走路時頭穩住",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources_from_battery(
    result: Mapping[str, Any], name: str, servo: str = "bam"
) -> list[dict[str, Any]]:
    """One source per delay of an A100 battery result (``gates`` from :func:`s2_battery.evaluate`)."""
    sha = (((result.get("meta") or {}).get("arms") or {}).get("cand") or {}).get("sha256")
    out = []
    for delay, gates in (result.get("gates") or {}).items():
        out.append({"kind": "a100", "servo": servo, "delay": delay, "file": name, "walk_sha256": sha,
                    "complete": bool(result.get("complete")), "gates": dict(gates)})  # fmt: skip
    return out


def sources_from_mac(
    summary: Mapping[str, Any], name: str, arm: str | None
) -> list[dict[str, Any]]:
    """One source per servo model × delay of a Mac summary's ``verdicts`` for ``arm``."""
    out = []
    for label, gates in (summary.get("verdicts") or {}).items():
        servo, delay, got = label.split("/", 2)
        if arm is not None and got != arm:
            continue
        out.append({"kind": "mac", "servo": servo, "delay": delay, "arm": got, "file": name,
                    "walk_sha256": (summary.get("walk_sha256") or {}).get(label), "gates": dict(gates)})  # fmt: skip
    return out


def gate_status(key: str, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A gate over all sources: ``passed`` by the rules of the module doc, and every verdict."""
    gate = C.GATE_BY_KEY[key]
    by = {f"{s['kind']}:{s['servo']}/{s['delay']}": (s["gates"].get(key) or {}).get("verdict", "NOT MEASURED")
          for s in sources}  # fmt: skip
    verdicts = list(by.values())
    if gate.kind == "report":
        return {"passed": True, "rule": "report", "by_source": by}
    if "FAIL" in verdicts:
        return {"passed": False, "rule": "failed somewhere", "by_source": by}
    if gate.kind in RATE_KINDS:
        ok = "PASS" in verdicts
        return {"passed": ok, "rule": "rate: PASS in a resolving source, no FAIL", "by_source": by}
    missing = [servo for servo in SERVOS
               if not any(v == "PASS" for s, v in zip(sources, verdicts, strict=True) if s["servo"] == servo)]  # fmt: skip
    return {
        "passed": not missing,
        "rule": "PASS on both servo models",
        "missing": missing,
        "by_source": by,
    }


def abilities(sources: Sequence[Mapping[str, Any]], stage: str, walk_sha: str) -> dict[str, Any]:
    stage_gates = {g.key for g in C.gates_for(stage)}
    groups: dict[str, Any] = {}
    why: list[str] = []
    for group, keys in GROUPS.items():
        keys = tuple(k for k in keys if k in stage_gates)
        status = {k: gate_status(k, sources) for k in keys}
        granted = bool(keys) and all(s["passed"] for s in status.values())
        groups[group] = {"granted": granted, "zh": GROUP_ZH[group], "gates": status}
        if not granted:
            short = [k for k, s in status.items() if not s["passed"]]
            why.append(f"{GROUP_ZH[group]}：還沒通過（{', '.join(short) or '這一階段沒有這項'}）")
    usable = groups["usable"]["granted"]
    pose_safe = usable and groups["pose_safe"]["granted"]
    pose = {p: bool(pose_safe and groups[p]["granted"]) for p in POSE_PROGRAMS}
    return {
        "schema": SCHEMA,
        "stage": stage,
        "walk_sha256": walk_sha,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "usable": usable,
        "pose": pose,
        "neck_standing": bool(usable and groups["neck_standing"]["granted"]),
        "gaze_stable": bool(usable and stage == "gaze" and groups["gaze_stable"]["granted"]),
        "groups": groups,
        "sources": [{k: v for k, v in s.items() if k != "gates"} for s in sources],
        "why_zh": why,
        "rules": __doc__.split("The rules")[1].split("``python -m")[0].strip() if __doc__ else "",
    }


def _check_sha(sources: Iterable[Mapping[str, Any]], walk_sha: str) -> list[str]:
    return [f"{s['kind']}:{s['servo']}/{s['delay']} evaluated {s['walk_sha256']}"
            for s in sources if s.get("walk_sha256") and s["walk_sha256"] != walk_sha]  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m duckmind.skilltrain.s2_abilities", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--walk", required=True, help="the exported ONNX (final/velstand_duck.onnx)")
    ap.add_argument("--stage", required=True, choices=C.STAGES)
    ap.add_argument(
        "--battery", action="append", default=[], help="an s2_battery results JSON (repeatable)"
    )
    ap.add_argument("--battery-servo", default="bam")
    ap.add_argument(
        "--mac", action="append", default=[], help="an s2_mac summary.json (repeatable)"
    )
    ap.add_argument("--arm", default=None, help="the Mac arm label of this network")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    walk = Path(args.walk)
    walk_sha = sha256(walk)
    sources: list[dict[str, Any]] = []
    for b in args.battery:
        sources += sources_from_battery(
            json.loads(Path(b).read_text()), Path(b).name, args.battery_servo
        )
    for m in args.mac:
        sources += sources_from_mac(json.loads(Path(m).read_text()), Path(m).name, args.arm)
    wrong = _check_sha(sources, walk_sha)
    if wrong:
        raise SystemExit("evaluations of another file: " + "; ".join(wrong))
    if not sources:
        raise SystemExit("no evaluations given")
    res = abilities(sources, args.stage, walk_sha)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    granted = [g for g in GROUPS if res["groups"][g]["granted"]]
    print(f"usable={res['usable']} granted={granted} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["GROUPS", "abilities", "gate_status", "main", "sources_from_battery", "sources_from_mac"]
