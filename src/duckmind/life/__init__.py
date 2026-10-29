"""One duck's life: the calendar clock it lives by, the record of who it is and the snapshot of
what it has learnt (docs/design/GROWTH-G1-2026-10-10.md; owner principle 2026-10-10: the neural duck
is the living face, the growth is the product's core, and the two must be one duck).

* :mod:`duckmind.life.clock` — :class:`DuckClock`, the one clock: real time by default, a speed
  factor for tests. The neural duck's day and night, the room's lights and Duck Core's growth
  days all read it.
* :mod:`duckmind.life.record` — :class:`LifeRecord`, the one duck record on disk (name, seed,
  birth, where its Duck Core save and its neural snapshot live).
* :mod:`duckmind.life.snapshot` — the versioned snapshot of the neural duck's slow state (what it
  learnt of places and people, habituation, drives) and how it is restored after a restart.
"""
