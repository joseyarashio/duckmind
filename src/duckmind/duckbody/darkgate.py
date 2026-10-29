"""The dark gate (spec §4.5 walking rules; owner C1, C5, 2026-09-28 ~12:30 「2. 放寬」 and
~10:45 「3. 一律放寬，即便跌倒才是真實模擬」): what a walk may do in this light.

Pure integer function :func:`allow_walk` over the light level duck-eye reports (with its 2-look
hysteresis) and whether the duck has a ToF. **No light refuses or shortens a walk**: since the
owner's 「2. 放寬」 (C2 reversed) the dark refuses nothing, and since the phase-4 landing
(2026-09-28, 「一律放寬」) it no longer caps a segment either — the 3 s dark segment was pacing
only, so it went. In a dark room the duck walks as it would in the light, and physics decides
whether it bumps or falls. What the dark still does is honest, not a rule — the eye simply sees
less there (dim frames claim 20 % less floor, dark and black frames none), so the free-cell
memory's :func:`~duckmind.duckbody.lookplan.look_veto` finds less of the route seen free and
steps slowly into the rest, as it does with any floor no look has seen:

========  ==========================================  ====================================
level     camera only                                 with ToF
========  ==========================================  ====================================
bright    walk (look_veto: cells seen ≤ 10 min)       same
dim       walk (look_veto: cells seen ≤ 5 min)        same
dark      walk (look_veto: the same seen cells;       same, plus the ToF reflexes
          unseen floor a slow 2 s step)
black     as dark                                     as dark
none      the eye is down or its level is stale:      as dark
          as black
========  ==========================================  ====================================

* **C1** (kept, a light-level fact) — in total darkness with only the dock's own light in view,
  duck-eye calls the frame ``dark`` (很暗), not ``black``.
* **C2** (reversed) — walking in the dark no longer needs the route seen lit within 30 min: the
  memory keeps what was seen free for as long as it would in the light (10 min bright, 5 min
  dim), and everything else is unseen floor.
* **C5** (kept, a label) — a backlight (or the dock's cyan cue) gives a bearing, never a range;
  a walk heading at it is reported ``toward_backlight``.

:func:`walk_gate` adapts it to walkexec's ``DarkGate`` hook ``(ctx, cmd) → (allow, max_ms,
why)`` (a stricter gate may still refuse or cap; this one never does).
"""

from __future__ import annotations

from dataclasses import dataclass

from duckmind.duckbody.protocol import ActContext, WalkCmd

TOWARD_LIGHT_MRAD = 150  # "toward the backlight": heading within this of its bearing
LIGHT_LEVELS = ("bright", "dim", "dark", "black", "none")


@dataclass(frozen=True)
class DarkVerdict:
    allow: bool
    max_ms: int | None  # a cap on the segment, ``None`` = no cap from the light (always, now)
    rule: str  # light / dark / toward_backlight
    reason: str | None = None  # the walk.done reason when refused (never, since 「2. 放寬」)


def allow_walk(
    level: str,
    *,
    has_tof: bool,
    backlight_heading_mrad: int | None = None,
) -> DarkVerdict:
    """May a walk start, and how long may its segment be? Always, as long as it asked (owner
    2026-09-28 「2. 放寬」 and 「一律放寬」): bright and dim are ``light``; dark, black and
    ``none`` (the eye down) are ``dark`` — or ``toward_backlight`` when the walk heads within
    :data:`TOWARD_LIGHT_MRAD` of a visible backlight (``backlight_heading_mrad``: the walk's
    heading error to it; ``None``: none in view; C5, a label only). ``has_tof`` does not change
    the answer (the ToF's own reflexes still stop the walk where it sees something)."""
    if level not in LIGHT_LEVELS:
        level = "none"
    if level in ("bright", "dim"):
        return DarkVerdict(True, None, "light")
    if backlight_heading_mrad is not None and abs(backlight_heading_mrad) <= TOWARD_LIGHT_MRAD:
        return DarkVerdict(True, None, "toward_backlight")
    return DarkVerdict(True, None, "dark")


def walk_gate(ctx: ActContext, cmd: WalkCmd) -> tuple[bool, int | None, str]:
    """walkexec's ``DarkGate``: a walk toward a just-seen light (``toward`` on a ``dock``/
    ``home`` walk) counts as toward the backlight; the route's cells are look_veto's."""
    heading = None
    if cmd.toward is not None and cmd.purpose in ("dock", "home", "nest"):
        heading = cmd.toward.b_mrad
    verdict = allow_walk(ctx.light, has_tof=ctx.depth is not None, backlight_heading_mrad=heading)
    return verdict.allow, verdict.max_ms, verdict.rule


__all__ = ["DarkVerdict", "LIGHT_LEVELS", "TOWARD_LIGHT_MRAD", "allow_walk", "walk_gate"]
