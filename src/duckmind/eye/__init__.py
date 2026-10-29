"""duck-eye: what the duck can tell from its own camera frames (spec §4, §6; phase 3: the process).

Perception code that runs the same on the board and beside the simulator, so it takes frames and
sensor readings only — never the simulator's truth — and it does not import
:mod:`duckmind.world`. duck-body talks to it over ``eye.sock`` (:mod:`duckmind.eye.protocol`,
:mod:`duckmind.eye.server`); Duck Core does not read it. The frame, the fd hand-over, the clock
bridge and the pose pairing are :mod:`duckmind.eye.frames`; K and the lens
:mod:`duckmind.eye.camera`; the light meter :mod:`duckmind.eye.light`; the look budget
:mod:`duckmind.eye.governor`; the tasks ``floorline``, ``nesttag``, ``things`` and ``detect``.
"""
