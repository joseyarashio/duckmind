"""Write duckmind's trained-motion table (``skills.json``) beside its networks.

    python -m duckmind.pins.motions FETCHED_DIR POLICY_DIR OUT_DIR

FETCHED_DIR holds the seven networks fetched from ``joseyarashio/duckmind-motion-skills`` (file names
as in :data:`duckmind.skilltrain.motion_table.FILES`); POLICY_DIR is Pollen's policy set v9, whose
``roulade.onnx`` is the flip trick and whose ``velstand.onnx`` / ``alpha_sitstand.onnx`` every skill
hands back to. OUT_DIR gets the eight networks and ``skills.json``, with the guard caps of
``motion-guard.json``. It is the table the live simulator runs on policy set v9; ``python -m
duckmind.setup_cmd`` and the public play image both write it with :func:`install`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from duckmind.pins import MOTION_GUARD_FILE
from duckmind.skilltrain import motion_table as MT


def install(fetched: Path, policies: Path, out: Path) -> tuple[Path | None, list[str]]:
    """Write the table into ``out``; returns (its path, []) or (None, what is wrong)."""
    fetched, policies = fetched.resolve(), policies.resolve()
    sources: dict[str, str | Path] = {
        role: fetched / name for role, name in MT.FILES.items() if role != "roulade"
    }
    sources["roulade"] = policies / "roulade.onnx"
    missing = [str(p) for p in sources.values() if not Path(p).is_file()]
    if missing:
        return None, [f"missing: {', '.join(missing)}"]
    if MT.sha256_of(sources["roulade"]) != MT.ROULADE_SHA256:
        return None, [f"{sources['roulade']}: not Pollen's roulade"]
    try:
        table = MT.install(out, sources, walk_dir=policies, base_table=MOTION_GUARD_FILE)
    except ValueError as err:
        return None, [str(err)]
    problems = MT.check(table)
    return (None, problems) if problems else (table, [])


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    table, problems = install(*(Path(a) for a in argv))
    if table is None:
        print("; ".join(problems), file=sys.stderr)
        return 1
    print(f"{table}: pins ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
