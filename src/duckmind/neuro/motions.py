"""「動作試玩」: every motion trained on Colab, as the page lists and plays it (2026-10-05).

The registry of what a button does: the table entry it starts (:mod:`duckmind.skilltrain.motion_table`),
how long a held entry is kept (the freezes, the turns, the lie-low), the posture the Mac gates
allowed it from, and the gate verdict the page shows (通過 / 有條件通過 / 未通過). Pure data; the page's
words are catalog keys ``page.motion.<id>.name`` / ``.limit`` (the server sends ids only), the refusals
are ``neuro.skill.reason.*`` messages (:mod:`duckmind.skilltrain.guard`).

Which of them also join the brain (``brain``, said in plain words on the page): the startles and freezes
by the size of the scare (2026-10-10: a startle of level 2 or more is answered by the hop-back, two small steps back, unless
the duck's own memory says the way behind is blocked or the scare came from behind; ``then``: the page's startle buttons of
level 2 to 4 do the same, the dip and then the steps), the shakes and the stretches by their acts, a greeting bow, a peekaboo lie-low, and the
on-the-spot turn only out of a corner (the escape that has failed three times at one place). The hop-over is for the page: nothing in the brain asks for it. (The march and the mat steps were taken out
of the table on 2026-10-09 to free slots.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from duckmind.skilltrain.guard import ABORT, BOW_RELEASE, RELEASE

VERDICT_LABEL = {"pass": "page.motion.verdict.pass", "limits": "page.motion.verdict.limits",
                 "fail": "page.motion.verdict.fail"}  # fmt: skip
GROUPS = ("fear", "comfort", "move", "play")


@dataclass(frozen=True)
class Motion:
    id: str  # the page's key (also the API's)
    group: str  # fear | comfort | move | play | mat
    entry: str  # the table entry that starts it
    family: str  # the gate family (motion_table.VERDICT)
    verdict: str  # pass | limits | fail
    postures: tuple[str, ...]  # where the page may start it (the gates')
    hold_s: float = 0.0  # a held entry is re-requested this long on the page …
    end: str | None = None  # … and ended by this entry
    brain: str = ""  # how the brain uses it ("": nothing does)
    max_walk_mps: float | None = None  # from a walk only up to this speed (the guard's rule)
    entries: tuple[str, ...] = ()  # every table entry this motion needs (default: entry and end)
    then: str = ""  # a table entry that follows it once the duck has stopped (the startles' two small steps back)

    def needs(self) -> tuple[str, ...]:
        extra = tuple(x for x in (self.entry, self.end, self.then) if x)
        return self.entries or extra


MOTIONS: tuple[Motion, ...] = (
    Motion(
        "startle_1",
        "fear",
        "startle_1",
        "startle",
        "pass",
        ("stand", "walk"),
        brain="startle",
        max_walk_mps=0.55,
    ),
    Motion(
        "startle_2",
        "fear",
        "startle_2",
        "startle",
        "pass",
        ("stand", "walk"),
        brain="startle",
        max_walk_mps=0.55,
        then="hop_back",
    ),
    Motion(
        "startle_3",
        "fear",
        "startle_3",
        "startle",
        "pass",
        ("stand", "walk"),
        brain="startle",
        max_walk_mps=0.55,
        then="hop_back",
    ),
    Motion(
        "startle_4",
        "fear",
        "startle_4",
        "startle",
        "pass",
        ("stand", "walk"),
        brain="startle",
        max_walk_mps=0.55,
        then="hop_back",
    ),
    Motion(
        "freeze_2s",
        "fear",
        "freeze_3",
        "freeze",
        "pass",
        ("stand", "walk"),
        hold_s=2.0,
        end=RELEASE,
        brain="freeze",
        max_walk_mps=0.55,
    ),
    Motion(
        "freeze_3s",
        "fear",
        "freeze_4",
        "freeze",
        "pass",
        ("stand", "walk"),
        hold_s=3.0,
        end=RELEASE,
        brain="freeze",
        max_walk_mps=0.40,
    ),
    Motion(
        "hop_back",
        "fear",
        "hop_back",
        "hop_back",
        "limits",
        ("stand",),
        brain="hop_back",
        max_walk_mps=0.40,
    ),
    Motion("shake_1", "comfort", "shake_1", "shake", "pass", ("stand",), brain="shake"),
    Motion("shake_2", "comfort", "shake_2", "shake", "pass", ("stand",), brain="shake"),
    Motion(
        "stretch_tall",
        "comfort",
        "stretch_tall",
        "stretch_tall",
        "pass",
        ("stand",),
        brain="stretch",
    ),
    Motion(
        "stretch_sit", "comfort", "stretch_sit", "stretch_sit", "pass", ("sit",), brain="stretch"
    ),
    Motion(
        "turn_left",
        "move",
        "turn_left",
        "turn",
        "limits",
        ("stand", "walk"),
        hold_s=2.0,
        end=ABORT,
        brain="corner",
        max_walk_mps=0.40,
    ),
    Motion(
        "turn_right",
        "move",
        "turn_right",
        "turn",
        "limits",
        ("stand", "walk"),
        hold_s=2.0,
        end=ABORT,
        brain="corner",
        max_walk_mps=0.40,
    ),
    Motion(
        "quick_turn_left",
        "move",
        "quick_turn_left",
        "quick_turn",
        "limits",
        ("stand", "walk"),
        hold_s=0.8,
        end=ABORT,
        brain="pivot",
        max_walk_mps=0.40,
    ),
    Motion(
        "quick_turn_right",
        "move",
        "quick_turn_right",
        "quick_turn",
        "limits",
        ("stand", "walk"),
        hold_s=0.8,
        end=ABORT,
        brain="pivot",
        max_walk_mps=0.40,
    ),
    Motion(
        "hop_over_2",
        "move",
        "hop_over_2",
        "hop_over",
        "limits",
        ("stand", "walk"),
        max_walk_mps=0.50,
    ),
    Motion(
        "hop_over_4",
        "move",
        "hop_over_4",
        "hop_over",
        "limits",
        ("stand", "walk"),
        max_walk_mps=0.50,
    ),
    Motion("bow", "play", "bow", "bow", "pass", ("stand",), brain="greet"),
    Motion(
        "lie_low",
        "play",
        "lie_low",
        "lie",
        "limits",
        ("stand",),
        hold_s=2.0,
        end=BOW_RELEASE,
        brain="peekaboo",
    ),
)
BY_ID = {m.id: m for m in MOTIONS}


def status(loaded: set[str] | frozenset[str] | None = None) -> list[dict[str, Any]]:
    """The page's list (``/api/neuro/status`` ``motions``): ids and facts, never words. ``loaded``:
    the table entries this robotd has (None: unknown — nothing is marked missing)."""
    out = []
    for m in MOTIONS:
        row: dict[str, Any] = {
            "id": m.id,
            "group": m.group,
            "verdict": m.verdict,
            "postures": list(m.postures),
            "brain": bool(m.brain),
            "held_s": m.hold_s or None,
            "needs": list(
                m.needs()
            ),  # the table entries it takes (the page greys it out without them)
        }
        if loaded is not None:
            row["installed"] = all(name in loaded for name in m.needs())
        out.append(row)
    return out


__all__ = ["BY_ID", "GROUPS", "MOTIONS", "Motion", "VERDICT_LABEL", "status"]
