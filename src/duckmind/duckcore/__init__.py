"""Duck Core: the duck's long-term self (Duck Core v3, first implementation).

This package holds the deterministic "self" logic from the v3 design: growth days,
seated expression choices, reactions, bounded learning, branch development, the
voice lineage and layered memory. It never talks to hardware. The simulator in
``duckmind.simweb`` drives it with synthetic stimuli; on the robot the same code is
meant to run inside ``duck-self`` behind ``duck-body``.
"""

from duckmind.duckcore.engine import DuckCore, HardwareConfig

__all__ = ["DuckCore", "HardwareConfig"]
