"""Curiosity P1: the Act Runner and its numbers (curiosity spec §5–§6, §13; physical-design spec
§15; phase 3, owner decisions A1–A6, 2026-09-27).

Duck-body side (pure, clock-injected, no world, no truth — they run on the board too):

* :mod:`~duckmind.curiosity.act_runner` — :class:`~duckmind.curiosity.act_runner.CurioStep`, one
  ``duck.curio`` step as a duck-body act executor (look, approach, bump, peck, nibble, tap, grab,
  carry, put down, kick, dribble);
* :mod:`~duckmind.curiosity.prims` — the motion primitives, ``AUTONOMOUS_SKILLS`` and the rules no
  curiosity act may break;
* :mod:`~duckmind.curiosity.perceive` — the adapter over duck-eye's ``things`` task and the small
  odometry-frame tracker;
* :mod:`~duckmind.curiosity.outcomes` — outcomes from perception, and the outcome tables;
* :mod:`~duckmind.curiosity.cone` — kick-cone-v1;
* :mod:`~duckmind.curiosity.calib` — the standoffs, windows and step lengths measured under the
  robotd path, and the patience tiers.

Simulator side (may import the world; never imported by the modules above):

* :mod:`~duckmind.curiosity.curio_calib` — turns measurement records into the calibration summary
  and the outcome table (the evidence files under ``docs/validation/evidence/phase3/``).
"""
