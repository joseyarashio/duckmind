"""The motor programs of a:play's and a:sound's acts, in :mod:`duckmind.neuro.repertoire`'s
builder form (a:integrate 2026-09-29: moved here from :mod:`duckmind.neuro.skill_play` and
:mod:`duckmind.neuro.skill_sound` unchanged, so :mod:`~duckmind.neuro.repertoire` can register them
without importing the brain; those modules re-export them).

* ``play_ball`` 「玩球」, ``carry_toy`` 「叼玩具回窩」, ``sunbathe`` 「曬太陽」 (a:play);
* ``theremin`` 「玩特雷門琴」, ``dance`` 「聽音樂跳舞」, ``wheee_ride`` 「咻～兜風」 (a:sound).

The social acts' programs are :mod:`duckmind.neuro.skill_social_programs`.
"""

from __future__ import annotations

from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import skill_ride as SR

# ── a:play ───────────────────────────────────────────────────────────────────


def build_play_ball(b: Any, ctx: Any) -> Any:
    if ctx.mode == "walk":
        b.to(b.u(250, 400), np=b.u(0.15, 0.3), hp=b.u(-0.35, -0.2))  # eyes on the ball ahead
        b.hold(b.u(500, 1100), drift=0.0)
        return b.score(label="追球", gaze=1.0, alive=0.5)
    b.to(b.u(250, 400), np=b.u(0.3, 0.5), hp=b.u(-0.6, -0.4))  # a good look down at it
    if b.p(0.4):
        b.rel(b.u(250, 380), dhr=b.sign() * b.u(0.3, 0.5))  # a little tilt: which way?
    b.hold(b.u(600, 1300), drift=0.0)
    return b.score(label="盯著球", gaze=0.9, alive=0.4)


def build_carry_toy(b: Any, ctx: Any) -> Any:
    if ctx.mode == "walk":
        b.to(b.u(300, 450), np=0.0, hp=b.u(0.0, 0.1), hy=0.0, hr=0.0)
        b.hold(b.u(1500, 2500), drift=0.0)
        return b.score(label="叼著走", gaze=0.1, alive=0.15)
    b.to(b.u(300, 450), np=b.u(0.2, 0.35), hp=b.u(-0.5, -0.3))
    b.hold(b.u(700, 1400), drift=0.0)
    return b.score(label="低頭看小玩具", gaze=0.8, alive=0.3)


def build_sunbathe(b: Any, ctx: Any) -> Any:
    if ctx.mode == "walk":
        b.to(b.u(350, 500), np=b.u(0.05, 0.15), hp=b.u(-0.2, -0.1))
        b.hold(b.u(900, 1500))
        return b.score(label="走向太陽", gaze=0.6, alive=0.5)
    # seated in the sun: the head sinks low and relaxed, the face a little up toward the light,
    # held long; now and then a slow drowsy dip (no eyelids on this body: the carriage says it)
    b.to(b.u(900, 1400), np=b.u(0.3, 0.45), hp=b.u(0.1, 0.25), hr=b.sign() * b.u(0.0, 0.15))
    b.hold(b.u(3500, 7000), sigma=0.35, drift=0.02)
    if b.p(0.6):
        b.rel(b.u(1100, 1500), dhp=-b.u(0.18, 0.28), ease="soft")
        b.hold(b.u(600, 1400), drift=0.0)
        b.rel(b.u(1300, 1800), dhp=b.u(0.15, 0.25), ease="soft")
    b.hold(b.u(2500, 5000), sigma=0.35, drift=0.02)
    if b.p(0.15):
        b.vocal(b.tag(("", "coo"), silent=2.0), 0.3)
    return b.score(label="瞇著眼曬太陽", gaze=0.15, alive=0.2)


PLAY_BUILDERS = {
    "play_ball": build_play_ball,
    "carry_toy": build_carry_toy,
    "sunbathe": build_sunbathe,
}
PLAY_REACTIVE = frozenset(PLAY_BUILDERS)

# ── a:sound ──────────────────────────────────────────────────────────────────


def build_theremin(b: Any, ctx: Any) -> Any:
    """Playing the theremin: it looks up at the hand and holds its head still there (the brain's
    gaze has the yaw); now and then a slow listening tilt. The beak is robotd's (the note's).
    Looking up matters to the instrument too: at HOME a standing duck's lowest ToF zones see the
    floor ~45 cm ahead (robotd's tracker plays the raw beam); with the nose up ~12–18° the floor
    moves past 58–69 cm and the hand plays the whole band (a:sound, headless ToF model)."""
    side = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    b.to(b.u(300, 450), hp=b.u(0.65, 0.9), np=b.u(-0.15, 0.0), hr=0.0, m=0.0)
    b.hold(b.u(1800, 2600), drift=0.01)
    if b.p(0.35):
        b.rel(b.u(350, 500), dhr=0.35 * side, ease="soft")
        b.hold(b.u(600, 1000), drift=0.0)
        b.rel(b.u(350, 500), dhr=-0.35 * side, ease="soft")
    b.hold(b.u(600, 1200), drift=0.01)
    return b.score(label="聽自己的聲音", alive=0.15, gaze=1.0)


def build_dance(b: Any, ctx: Any) -> Any:
    """Dancing: a lifted, open carriage held (the beat's nod and sway are the motor's overlay,
    phase-locked to what it hears)."""
    b.to(b.u(250, 400), hp=b.u(0.15, 0.3), np=b.u(-0.15, 0.0), hy=0.0, hr=0.0, m=0.0)
    b.hold(b.u(2500, 4000), drift=0.0)
    return b.score(label="跟著拍子點頭", alive=0.1, gaze=0.2)


def build_wheee_ride(b: Any, ctx: Any) -> Any:
    """In the air, thrilled: the head up into the wind, the beak a little open."""
    pose = SR.JOY_POSE
    b.to(b.u(250, 400), np=pose.np, hp=pose.hp, hy=0.0, hr=0.0, m=pose.mouth)
    b.hold(b.u(1500, 2500), drift=0.03)
    return b.score(label=msg("neuro.vocal.wheee"), alive=0.2, gaze=0.2)


SOUND_BUILDERS = {"theremin": build_theremin, "dance": build_dance, "wheee_ride": build_wheee_ride}

BUILDERS = {**PLAY_BUILDERS, **SOUND_BUILDERS}

__all__ = [
    "BUILDERS",
    "PLAY_BUILDERS",
    "PLAY_REACTIVE",
    "SOUND_BUILDERS",
    "build_carry_toy",
    "build_dance",
    "build_play_ball",
    "build_sunbathe",
    "build_theremin",
    "build_wheee_ride",
]
