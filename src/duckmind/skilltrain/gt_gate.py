"""tr:gate: the Mac gate runner with a servo command delay on the ``mjcf`` profile.

The ``mjcf`` profile (:mod:`~duckmind.skilltrain.servo`) stands for the owner's live simulator,
whose lockstep body server (``duckmind.bodyhost``) applies robotd's joint targets on the very next
physics step: **no command delay**. Training delays them 15–30 ms (the velocity recipe's DR, 3–6
physics steps of 5 ms), and so does the ``bam`` profile here. The Stage-1 skill networks turned
out to depend on that delay (tr:gate 2026-09-28, ``gate/lag_probe.json``: hop_back on a nominal
duck falls 10/10 with 0 ms and 0/10 with 20 ms on both servo models; the shake is under its gate
with 0 ms and passes with 20 ms).

This wrapper runs :mod:`duckmind.skilltrain.gate` unchanged, with one extra option for ``run``::

    python -m duckmind.skilltrain.gt_gate run --command-delay dr|none|<steps> [gate run args]

* ``dr`` — each ``mjcf`` duck gets the delay its ``bam`` twin draws (the recipe's 3–6 steps; the
  same seed is the same duck), so the ``mjcf`` column is judged with training's latency;
* ``<steps>`` — a fixed delay in physics steps (4 = 20 ms, the proposal for our live simulator);
* ``none`` — no delay (the profile as it is; the same as ``gate run``).

``bam`` ducks keep their own drawn delay whatever the option says. The delay each duck ran with
is recorded in its row (``duck.lag_steps``). Other sub-commands pass straight through.
"""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Sequence
from typing import Any

from duckmind.skilltrain import dr as DR
from duckmind.skilltrain import gate as G
from duckmind.skilltrain import programs as P
from duckmind.skilltrain.skillduck import SkillDuck

PHYSICS_STEP_MS = 5.0


def delay_steps(option: str, seed: int) -> int:
    """The command delay (physics steps) for an ``mjcf`` duck of ``seed`` under ``option``."""
    if option == "none":
        return 0
    if option == "dr":
        return int(DR.draw(seed, "recipe").lag_steps)
    steps = int(option)
    if steps < 0:
        raise ValueError("a command delay cannot be negative")
    return steps


def delayed_duck_class(option: str) -> type[SkillDuck]:
    """A :class:`SkillDuck` whose ``mjcf`` servos see robotd's targets ``option`` late."""

    class DelayedDuck(SkillDuck):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.servo_name == "mjcf":
                self.lag_steps = delay_steps(option, int(self.draw.seed))
                self._lag = deque(maxlen=self.lag_steps + 1)

    DelayedDuck.__name__ = f"DelayedDuck[{option}]"
    return DelayedDuck


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    option = "none"
    if "--command-delay" in args:
        i = args.index("--command-delay")
        option = args[i + 1]
        del args[i : i + 2]
        delay_steps(option, 0)  # validate early
    if args and args[0] == "run":
        if option != "none":
            P.SkillDuck = delayed_duck_class(option)  # type: ignore[misc]
        _note(args, option)
    return G.main(args)


def _note(args: list[str], option: str) -> None:
    """``OUT/delay-<servo>.json``: which command delay this run's ``mjcf`` ducks had."""
    import json
    from pathlib import Path

    def value(flag: str) -> str | None:
        return args[args.index(flag) + 1] if flag in args else None

    out, servo = value("--out"), value("--servo")
    if out is None or servo is None:
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    doc = {
        "servo": servo,
        "command_delay": option,
        "applies_to": "mjcf ducks (bam ducks keep their drawn 3-6 steps)",
        "physics_step_ms": PHYSICS_STEP_MS,
    }
    (Path(out) / f"delay-{servo}.json").write_text(json.dumps(doc, indent=1))


__all__ = ["PHYSICS_STEP_MS", "delay_steps", "delayed_duck_class", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
