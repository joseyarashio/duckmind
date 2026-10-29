"""The motor programs of a:social's acts — in :mod:`duckmind.neuro.repertoire`'s builder form.

Each builder draws one unit of the act from the bout's seed, the duck's style and the command
(the same ``_B`` / ``Ctx`` API as every repertoire program: head poses in clip units, beak, calls,
posture cues). The brain's command carries what the act is about: ``cmd.target`` (its mother, the
shelter's way out), ``cmd.side`` (the loop's side, where it peeks), ``cmd.intensity``. The motor
overlays the brain's gaze (its eyes on its mother while it follows), the reflexes, liveliness and
the envelopes, as for every act; the unit's gesture is grown to be seen (``repertoire._visible``).

* ``follow_mom`` 「跟著媽媽走」 — walking: head up, eyes on her (the gaze pursues her), a peep now
  and then — more when she is out of sight (a duckling's contact call); stopped: it looks up at
  her and peeps.
* ``settle_feet`` 「窩在媽媽腳邊」 — it sits (posture cue), leans its head low toward her feet,
  now and then looks up at her, a soft coo.
* ``hide`` 「躲起來」 — running to the shelter: head low and forward, silent; in the shelter: it sits
  low (posture cue) with the head pulled in and down, very still.
* ``check_mom`` 「看一眼媽媽」 (a moment, fired by an imprinted duck whose mother has been out of view
  a few seconds): a quick turn of the head toward where it believes she is, a hold, and back.
* ``peek_out`` 「探頭看看」 (a moment, fired by the hide run as its fear ebbs): from the tucked pose
  the head comes up and turns toward the way out (``cmd.side`` × how far), holds, looks a little
  further, and sinks back.
* ``circle_settle`` 「睡前轉一圈」 — walking its loop: the head turned into the turn and a little
  down (looking at the spot it will lie on); stopped: a look down at the spot, settling.

Registered by :mod:`duckmind.neuro.skill_social_brain` (``BUILDERS``, ``REACTIVE``); the
integration adds them to ``repertoire.BUILDERS`` directly.
"""

from __future__ import annotations

from typing import Any

from duckmind.i18n import msg
from duckmind.neuro import repertoire as P

Score = P.Score


def build_follow_mom(b: Any, ctx: Any) -> Score:
    target = ctx.cmd.target
    lost = target is None
    if ctx.mode == "walk":
        if lost:  # she is out of sight: it looks about as it goes, peeping for her
            s = b.sign()
            b.to(b.u(350, 550), hy=s * b.u(0.4, 0.75), hp=b.u(0.15, 0.4), ease="snap")
            b.hold(b.u(400, 800))
            b.to(b.u(350, 550), hy=-s * b.u(0.3, 0.6))
            if b.p(0.7):
                b.to(150, m=0.7, ease="snap").vocal("chirp", 0.65).to(200, m=0.0)
            b.hold(b.u(300, 600))
            return b.score(label="找媽媽", gaze=0.3)
        b.to(b.u(300, 500), hp=b.u(0.25, 0.5), np=b.u(-0.15, 0.05))
        if b.p(0.25):
            b.to(140, m=0.6, ease="snap").vocal(b.tag(("chirp", "inquire"), silent=0.3), 0.5)
            b.to(200, m=0.0)
        b.hold(b.u(800, 1500))
        return b.score(label=msg("neuro.act.follow_mom.name"), gaze=1.0)
    b.to(b.u(250, 400), hp=b.u(0.45, 0.75), np=-0.2)
    if b.p(0.6 if lost else 0.35):
        b.to(160, m=0.7, ease="snap").vocal("chirp", 0.6).to(220, m=0.0)
    b.hold(b.u(700, 1400))
    return b.score(label="抬頭找媽媽" if lost else "抬頭看媽媽", gaze=1.0 if not lost else 0.4)


def build_settle_feet(b: Any, ctx: Any) -> Score:
    b.posture("sit")
    toward = 0.0
    if ctx.cmd.target is not None:
        toward = P.S.clamp(ctx.cmd.target.bearing / 0.6, -1.0, 1.0)
    if ctx.unit == 0:
        b.to(b.u(900, 1400), np=b.u(0.35, 0.55), hp=b.u(-0.3, -0.1), hy=0.7 * toward)
        b.hold(b.u(900, 1600), drift=0.01)
        b.vocal(b.tag(("coo", ""), silent=0.4), 0.3)
    else:
        b.hold(b.u(2500, 6000), drift=0.02)
        if b.p(0.55):  # a look up at her, then back down
            b.to(b.u(450, 700), np=-0.1, hp=b.u(0.45, 0.7), hy=0.9 * toward)
            b.hold(b.u(700, 1400))
            if b.p(0.35):
                b.vocal(b.tag(("coo", "chirp"), silent=0.5), 0.3)
            b.to(b.u(600, 900), np=b.u(0.3, 0.5), hp=b.u(-0.3, -0.1), hy=0.7 * toward)
    b.breathe(0.03, 0.25)
    return b.score(label=msg("neuro.act.settle_feet.name"), alive=0.35, gaze=0.5)


def build_hide(b: Any, ctx: Any) -> Score:
    if ctx.mode == "walk":
        b.to(b.u(200, 300), np=b.u(0.2, 0.35), hp=b.u(-0.15, 0.0), hy=0.0, ease="snap")
        b.hold(b.u(700, 1300), drift=0.01)
        return b.score(label="跑去躲", alive=0.2, gaze=0.2)
    target = ctx.cmd.target
    if target is not None and (target.range_m or 0.0) > 0.35:
        # a pause on the way (the gait's own stop): a quick look back, no sitting down yet
        s = b.sign()
        b.to(b.u(250, 350), hy=s * b.u(0.6, 0.85), hp=0.15, ease="snap").hold(b.u(300, 600))
        b.to(b.u(250, 350), hy=0.0, hp=0.05)
        return b.score(label="跑去躲（回頭看）", alive=0.2, gaze=0.2)
    b.posture("sit")
    b.to(b.u(500, 800), np=b.u(0.55, 0.8), hp=b.u(-0.6, -0.4), hy=0.1 * b.g(), m=0.0)
    b.hold(b.u(3000, 6000), drift=0.0)
    return b.score(label="躲著不動", alive=0.1, gaze=0.0)


def build_peek_out(b: Any, ctx: Any) -> Score:
    side = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    reach = 0.6 + 0.4 * P.S.clamp(ctx.cmd.intensity, 0.0, 1.0)
    b.to(b.u(350, 550), np=b.u(-0.2, 0.05), hp=b.u(0.25, 0.45), hy=side * reach * b.u(0.7, 0.9))
    b.hold(b.u(900, 1600), drift=0.0)
    if b.p(0.5):  # a little further out
        b.to(b.u(300, 450), hy=side * min(1.0, reach * b.u(0.95, 1.1)), hp=b.u(0.3, 0.5))
        b.hold(b.u(500, 1000), drift=0.0)
    b.to(b.u(500, 800), np=b.u(0.55, 0.75), hp=b.u(-0.55, -0.35), hy=0.0)
    return b.score(label=msg("neuro.act.peek_out.name"), alive=0.1, gaze=0.0)


def build_check_mom(b: Any, ctx: Any) -> Score:
    side = ctx.cmd.side if ctx.cmd.side in (-1, 1) else ctx.style.side
    yaw = side * P.S.clamp(ctx.cmd.intensity / 0.6, 0.35, 1.0)  # clip units: 0.6 rad each
    b.to(b.u(250, 380), hy=yaw, hp=b.u(0.1, 0.3), ease="snap")
    b.hold(b.u(600, 1000), drift=0.0)
    b.settle(b.u(400, 600))
    return b.score(label=msg("neuro.act.check_mom.name"), alive=0.3, gaze=0.0)


def build_circle_settle(b: Any, ctx: Any) -> Score:
    turning = ctx.cmd.vyaw if abs(ctx.cmd.vyaw) > 0.05 else float(ctx.cmd.side)
    s = 1 if turning > 0 else -1
    if ctx.mode == "walk":
        b.to(b.u(400, 600), hy=s * b.u(0.45, 0.7), hp=b.u(-0.35, -0.15), np=b.u(0.1, 0.25))
        b.hold(b.u(900, 1600), drift=0.01)
        if b.p(0.2):
            b.vocal(b.tag(("", "coo"), silent=1.5), 0.25)
        return b.score(label=msg("neuro.act.circle_settle.name"), gaze=0.2, alive=0.4)
    b.to(b.u(450, 700), np=b.u(0.4, 0.6), hp=b.u(-0.7, -0.45), hy=s * 0.2)
    b.hold(b.u(700, 1200), drift=0.01)
    b.settle(b.u(500, 800))
    return b.score(label="看看要睡的地方", gaze=0.2, alive=0.4)


BUILDERS = {
    "follow_mom": build_follow_mom,
    "settle_feet": build_settle_feet,
    "hide": build_hide,
    "peek_out": build_peek_out,
    "check_mom": build_check_mom,
    "circle_settle": build_circle_settle,
}
# reactive: rebuilt when walking ↔ stopped changes (repertoire.mode_of's default rule)
REACTIVE = frozenset({"follow_mom", "hide", "circle_settle"})
# the moment programs a page names (repertoire.MOMENT_PROGRAMS: commanded at ≥ MOMENT_DEG)
MOMENT_PROGRAMS = frozenset({"peek_out", "check_mom"})

__all__ = ["BUILDERS", "MOMENT_PROGRAMS", "REACTIVE"]
