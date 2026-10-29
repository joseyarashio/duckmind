"""「用眼睛做」: a curiosity step for the simulator page's card 09 (owner decision A6, phase 3).

The existing 示範 buttons (``simweb/demo.py``) put the thing on the trained spot by truth. This
button asks duck-body for a real ``duck.curio`` step instead: the duck looks with its own eyes,
walks up, aligns and acts, and the page shows what the duck itself reported. Two helpers for the
page's bridge:

* :func:`eye_demo_command` — the ``duck.curio`` step for a layout thing (``home-layout-v1``
  ``things`` entry) and a verb, with the thing's size and whether it may be carried taken from
  the layout (what the family registered);
* :func:`describe_zh` — the owner-facing sentence (Traditional Chinese, plain words) for the
  ``body.curio_step`` the step answered with.

Pure: standard library and the duck-body protocol only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from duckmind.duckbody.protocol import CurioCmd
from duckmind.i18n import join, msg

VERBS_ZH = {
    "look": msg("curiosity.demo.VERBS_ZH.look"),
    "bump": msg("curiosity.demo.VERBS_ZH.bump"),
    "peck": msg("curiosity.demo.VERBS_ZH.peck"),
    "nibble": msg("curiosity.demo.VERBS_ZH.nibble"),
    "tap": msg("curiosity.demo.VERBS_ZH.tap"),
    "grab": msg("curiosity.demo.VERBS_ZH.grab"),
    "carry": msg("curiosity.demo.VERBS_ZH.carry"),
    "put_down": msg("curiosity.demo.VERBS_ZH.put_down"),
    "kick": msg("curiosity.demo.VERBS_ZH.kick"),
    "dribble": msg("curiosity.demo.VERBS_ZH.dribble"),
}
OUTCOME_ZH = {
    "seen": msg("curiosity.demo.OUTCOME_ZH.seen"),
    "lost": msg("curiosity.demo.OUTCOME_ZH.lost"),
    "moved": msg("curiosity.demo.OUTCOME_ZH.moved"),
    "rolled": msg("curiosity.demo.OUTCOME_ZH.rolled"),
    "tipped": msg("curiosity.demo.OUTCOME_ZH.tipped"),
    "stuck": msg("curiosity.demo.OUTCOME_ZH.stuck"),
    "nothing": msg("curiosity.demo.OUTCOME_ZH.nothing"),
    "missed": msg("curiosity.demo.OUTCOME_ZH.missed"),
    "in_beak": msg("curiosity.demo.OUTCOME_ZH.in_beak"),
    "pressed": msg("curiosity.demo.OUTCOME_ZH.pressed"),
    "squirted": msg("curiosity.demo.OUTCOME_ZH.squirted"),
    "unreachable": msg("curiosity.demo.OUTCOME_ZH.unreachable"),
    "held": msg("curiosity.demo.OUTCOME_ZH.held"),
    "slipped": msg("curiosity.demo.OUTCOME_ZH.slipped"),
    "tethered": msg("curiosity.demo.OUTCOME_ZH.tethered"),
    "placed": msg("curiosity.demo.OUTCOME_ZH.placed"),
}
ENDED_ZH = {
    "gave_up": msg("curiosity.demo.ENDED_ZH.gave_up"),
    "cone_blocked": msg("curiosity.demo.ENDED_ZH.cone_blocked"),
    "dark": msg("curiosity.demo.ENDED_ZH.dark"),
    "not_allowed": msg("curiosity.demo.ENDED_ZH.not_allowed"),
    "lifted": msg("curiosity.demo.ENDED_ZH.lifted"),
    "fell": msg("curiosity.demo.ENDED_ZH.fell"),
    "preempt": msg("curiosity.demo.ENDED_ZH.preempt"),
    "deadline": msg("curiosity.demo.ENDED_ZH.deadline"),
    "lost": msg("curiosity.demo.ENDED_ZH.lost"),
}


def _size(thing: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """(height, depth) in mm of a layout thing (a sphere, a cylinder, a box or parts)."""
    if "radius_mm" in thing and thing.get("shape") == "sphere":
        d = 2 * int(thing["radius_mm"])
        return d, d
    if thing.get("shape") == "cylinder":
        return int(thing.get("height_mm", 0)) or None, 2 * int(thing.get("radius_mm", 0)) or None
    size = thing.get("size_mm")
    if isinstance(size, (list, tuple)) and len(size) == 3:
        return int(size[2]), int(min(size[0], size[1]))
    parts = thing.get("parts")
    if isinstance(parts, list) and parts:
        top = 0
        thin = None
        for part in parts:
            psize = part.get("size_mm") or [0, 0, 0]
            offset = (part.get("offset_mm") or [0, 0, 0])[2]
            top = max(top, int(offset) + int(psize[2]))
            if int(offset) > 0:  # the fin / handle standing on the body: what the beak grips
                thin = int(min(psize[0], psize[1]))
        return top or None, thin
    return None, None


def eye_demo_command(
    thing: Mapping[str, Any],
    verb: str,
    *,
    act_id: str,
    deadline_ms: int,
    step: int = 1,
    tier: str = "normal",
    label: str | None = None,
) -> CurioCmd:
    """The ``duck.curio`` step for this layout thing. ``label`` is what duck-eye calls it: the
    thing's id once the family registered it with the eye (the simulated family pre-registers
    the toy kit, curiosity spec §9), else its colour name as duck-eye says it."""
    verbs = thing.get("verbs") or []
    if verb not in VERBS_ZH:
        raise ValueError(f"verb {verb!r} not in {tuple(VERBS_ZH)}")
    if verbs and verb not in verbs and verb not in ("look", "nibble", "put_down", "carry"):
        raise ValueError(f"{thing.get('id')}: {verb} is not one of its verbs {verbs}")
    h, d = _size(thing)
    params: dict[str, Any] = {
        "verb": verb,
        "label": label or str(thing.get("id")),
        "cls": str(thing.get("cls", "unknown")),
        "tier": tier,
        "carry_ok": thing.get("cls") == "handle_toy" and ("carry" in verbs or "grab" in verbs),
    }
    if h is not None:
        params["h_mm"] = h
    if d is not None:
        params["d_mm"] = d
    return CurioCmd(
        act_id=act_id,
        deadline_ms=deadline_ms,
        phase="step",
        thing=str(thing.get("id")),
        step=step,
        params=params,
    )


def describe_zh(
    event: Mapping[str, Any], thing_zh: str = msg("curiosity.demo.describe_zh.5fe1cf67")
) -> str:
    """One plain sentence for the owner about a ``body.curio_step``."""
    verb = VERBS_ZH.get(str(event.get("verb")), str(event.get("verb")))
    ended = str(event.get("ended_by", "done"))
    if ended != "done" and ended in ENDED_ZH:
        return msg(
            "curiosity.demo.describe_zh.0bf80d52",
            verb=verb,
            thing_zh=thing_zh,
            ENDED_ZH2=ENDED_ZH[ended],
        )
    outcome = OUTCOME_ZH.get(str(event.get("outcome")), str(event.get("outcome")))
    parts = [
        msg("curiosity.demo.describe_zh.b50d9516", thing_zh=thing_zh, verb=verb, outcome=outcome)
    ]
    moved = event.get("moved_mm")
    if (
        isinstance(moved, int)
        and moved >= 15
        and event.get("verb") in ("kick", "bump", "peck", "nibble", "dribble")
    ):
        parts.append(msg("curiosity.demo.describe_zh.453ab62f", p=max(1, round(moved / 10))))
    carried = event.get("carried_mm")
    if isinstance(carried, int) and carried > 0:
        parts.append(msg("curiosity.demo.describe_zh.797c6c55", p=max(1, round(carried / 10))))
    seconds = event.get("ms")
    if isinstance(seconds, int):
        parts.append(msg("curiosity.demo.describe_zh.e19de210", p=max(1, round(seconds / 1000))))
    return msg("curiosity.demo.describe_zh.828d40b0", text=join(*parts))


__all__ = ["ENDED_ZH", "OUTCOME_ZH", "VERBS_ZH", "describe_zh", "eye_demo_command"]
