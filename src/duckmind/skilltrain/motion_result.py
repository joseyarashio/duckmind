"""What a motion the page started really did (2026-10-05): the numbers and one plain line.

Duck-typed on the witness's ``Tick`` (``t_ms``, ``policy``, ``trunk`` = x, y, z, heading from the physics
truth with ``trunk_t``, ``tilt_deg``, ``fallen``, ``head``): the same samples :mod:`live_judge` reads. The
page says it as 「做完了」 / 「沒有做」 with what the body measurably did; the evaluators read ``numbers``
(did it run, did it fall, did it end standing, how long it took).

* the startles, freezes, the hop-back, the shakes and the stretches are judged by the Mac gates' own
  detectors (:func:`live_judge.judge_window`: the neck pull-back, the trunk dip, one crouch, the stop,
  the 6 cm back, the head swing, the rise);
* the others say what they measure: a turn the degrees turned, a march the travel it made, a bow or a
  lie-low how far the trunk tipped and sank, a hop-over or a mat step how far it went and how high it got.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from duckmind.i18n import msg
from duckmind.skilltrain import live_judge as J

END_STILL_S = 0.5  # the run is over once the label has been a plain posture this long
MAX_RUN_S = 14.0
FALL_TILT_DEG = 45.0
PLAIN = ("stand", "walk", "sit", "rise")


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def is_skill_label(policy: str | None) -> bool:
    return J.skill_of(policy) is not None


def judge_reflex(family: str) -> str | None:
    """The reflex name :func:`live_judge.judge_window` needs to grade this family."""
    if family in ("startle", "freeze", "hop_back"):
        return "startle"
    if family == "shake":
        return "body_shake"
    if family in ("stretch_tall", "stretch_sit"):
        return "stretch_yawn"
    return None


def summarise(
    ticks: Sequence[Any], t0_ms: int, entry: str, family: str, *, started_ms: int | None = None
) -> dict[str, Any]:
    """The run's numbers and its line. ``ticks``: from a little before the request until the end;
    ``t0_ms``: the request; ``started_ms``: the first tick robotd's label showed the entry."""
    skilled = [k for k in ticks if is_skill_label(getattr(k, "policy", ""))]
    ran = bool(skilled)
    first = int(skilled[0].t_ms) if skilled else None
    last = int(skilled[-1].t_ms) if skilled else None
    seconds = None if first is None or last is None else round((last - first) / 1000.0, 2)
    fell = any(bool(getattr(k, "fallen", False)) for k in ticks if int(k.t_ms) >= t0_ms)
    tilts = [float(k.tilt_deg) for k in ticks if getattr(k, "tilt_deg", None) is not None]
    tail = [k for k in ticks if last is not None and int(k.t_ms) > last]
    end_label = (tail[-1].policy if tail else ticks[-1].policy if ticks else "") or ""
    end_base = end_label.split(":", 1)[0]
    standing = (not fell) and end_base in ("stand", "walk")
    sitting = (not fell) and end_base == "sit"
    numbers: dict[str, Any] = {
        "ran": ran,
        "fell": fell,
        "ended_standing": standing,
        "ended_sitting": sitting,
        "seconds": seconds,
        "tilt_max_deg": None if not tilts else round(max(tilts), 1),
    }
    pts = [
        (int(k.t_ms), *[float(v) for v in k.trunk])
        for k in ticks
        if getattr(k, "trunk", None) is not None and int(k.t_ms) >= t0_ms - 300
    ]
    base = next((p for p in pts if p[0] >= (first if first is not None else t0_ms) - 100), None)
    line: Any = None
    verdict: str | None = None
    reflex = judge_reflex(family)
    if ran and reflex is not None:
        judged = J.judge_window(ticks, reflex, t0_ms)
        if judged is not None and judged.get("verdict") is not None:
            line, verdict = judged["zh"], judged["verdict"]
            for key in ("dip_mm", "deepest_mm", "neck_back_deg", "back_cm", "turn_deg", "rise_mm",
                        "head_yaw_rad", "still_s", "bob_mm"):  # fmt: skip
                if key in judged:
                    numbers[key] = judged[key]
    if ran and base is not None and line is None:
        # the heading and the position the run changed (the truth: for the page's words and the evaluators)
        end = max((p for p in pts if first is not None and p[0] <= (last or 0) + 700), default=None,
                  key=lambda p: p[0])  # fmt: skip
        if end is not None:
            dist = math.hypot(end[1] - base[1], end[2] - base[2])
            numbers["travel_cm"] = round(dist * 100, 1)
            numbers["dz_mm"] = round((end[3] - base[3]) * 1000, 1)
            turned = 0.0
            prev = base[4]
            for p in pts:
                if base[0] <= p[0] <= end[0]:
                    turned += _wrap(p[4] - prev)
                    prev = p[4]
            numbers["turned_deg"] = round(math.degrees(turned), 0)
            low = min((p[3] for p in pts if base[0] <= p[0] <= end[0]), default=base[3])
            numbers["sink_mm"] = round((base[3] - low) * 1000, 1)
        if family in ("turn", "quick_turn"):
            line = msg("neuro.motion.result.turned", deg=abs(int(numbers.get("turned_deg") or 0)),
                       secs=seconds or 0.0)  # fmt: skip
        elif family == "march":
            line = msg("neuro.motion.result.marched", secs=seconds or 0.0,
                       cm=round(numbers.get("travel_cm") or 0.0))  # fmt: skip
        elif family in ("bow", "lie"):
            line = msg("neuro.motion.result.bowed", deg=round(numbers["tilt_max_deg"] or 0),
                       mm=round(numbers.get("sink_mm") or 0.0))  # fmt: skip
        elif family in ("hop_over", "step_up", "step_down"):
            line = msg("neuro.motion.result.stepped", cm=round(numbers.get("travel_cm") or 0.0),
                       mm=abs(round(numbers.get("dz_mm") or 0.0)))  # fmt: skip
        else:
            line = msg("neuro.motion.result.did", secs=seconds or 0.0)
    if line is None:
        line = msg("neuro.motion.result.not_run")
    tail_msg = (
        msg("neuro.motion.result.fell") if fell
        else msg("neuro.motion.result.stood") if standing
        else msg("neuro.motion.result.sat") if sitting
        else msg("neuro.motion.result.unsure")
    )  # fmt: skip
    return {"verdict": verdict, "numbers": numbers, "zh": line, "tail_zh": tail_msg}


__all__ = ["END_STILL_S", "MAX_RUN_S", "PLAIN", "is_skill_label", "judge_reflex", "summarise"]
