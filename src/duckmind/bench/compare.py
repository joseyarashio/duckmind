"""Two source trees on the same clicks: the paired comparison, with a stopping rule.

A tree is a git ref of this repository (its ``src`` is exported once into the comparison's folder) or a snapshot
directory that has ``src/duckmind``. Each tree answers the clicks in its own interpreter with its own ``src`` first on
``PYTHONPATH``; the bench that asks and scores is always this one (``_boot.py``), so only the duck differs.

With ``looks`` > 1 the blocks are answered in that many equal parts and :func:`duckmind.bench.stats.sequential` is read
after each on the same-room clicks (the measure every go-there decision has used): the comparison ends at the first look
that decides it.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from duckmind.bench import report as RP
from duckmind.bench import runner as RN
from duckmind.bench import stats as S

REPO = Path(__file__).resolve().parents[3]


def tree_of(spec: str, work: Path) -> tuple[Path, str]:
    """``spec`` as (the ``src`` folder to put on PYTHONPATH, a label for the rows)."""
    p = Path(spec).expanduser()
    if (p / "src" / "duckmind").is_dir():
        return (p / "src").resolve(), p.name
    if (p / "duckmind").is_dir():
        return p.resolve(), p.parent.name
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short=10", f"{spec}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dest = work / "trees" / sha
    if not (dest / "src" / "duckmind").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(
            ["git", "-C", str(REPO), "archive", sha, "src"], capture_output=True, check=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(dest, filter="data")
    return dest / "src", f"{spec}@{sha}"


def run_tree(
    src: Path, label: str, out: Path, args: Sequence[str], log: Any = print
) -> list[dict[str, Any]]:
    """Answer clicks with the tree at ``src`` (``args``: the ``gothere run`` options that choose clicks, arm and body)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [*RN.main_script(), "gothere", "run", "--out", str(out), "--tree-label", label, *args]
    if log is not None:
        log(f"[{label}] {' '.join(cmd[2:])}")
    subprocess.run(cmd, env=env, check=True, stdout=sys.stderr)
    return RN.load_rows(out)


def split_looks(blocks: Sequence[int], looks: int) -> list[list[int]]:
    """``blocks`` in ``looks`` nearly equal consecutive parts (fewer when there are fewer blocks)."""
    looks = max(1, min(int(looks), len(blocks)))
    size, extra = divmod(len(blocks), looks)
    out, at = [], 0
    for i in range(looks):
        n = size + (1 if i < extra else 0)
        out.append(list(blocks[at : at + n]))
        at += n
    return out


def same_room_look(
    a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]
) -> tuple[int, int, int]:
    pairs = [(x, y) for x, y in RP.pair(a, b) if x.get("cls") == "same"]
    _, only_a, only_b, _ = RP.counts(pairs)
    return len(pairs), only_a, only_b


def compare(
    a: str,
    b: str,
    work: Path,
    *,
    set_name: str,
    blocks: Sequence[int],
    looks: int = 1,
    delta: float = 0.05,
    arm_a: str = "main",
    arm_b: str = "main",
    run_args: Sequence[str] = (),
    log: Any = print,
) -> dict[str, Any]:
    """Run the comparison (resumable: rows already in ``work`` are kept) and return its tables and verdict."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (src_a, label_a), (src_b, label_b) = tree_of(a, work), tree_of(b, work)
    out_a, out_b = work / "rows-a.jsonl", work / "rows-b.jsonl"
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    seen: list[tuple[int, int, int]] = []
    verdicts: list[S.Look] = []
    parts = split_looks(list(blocks), looks)
    done: list[int] = []
    for part in parts:
        done += part
        chosen = ["--set", set_name, "--blocks", ",".join(str(x) for x in done), *run_args]
        rows_a = run_tree(src_a, label_a, out_a, [*chosen, "--arm", arm_a], log)
        rows_b = run_tree(src_b, label_b, out_b, [*chosen, "--arm", arm_b], log)
        seen.append(same_room_look(rows_a, rows_b))
        verdicts = S.sequential(seen, len(parts), delta)
        if log is not None:
            log(f"look {len(seen)}/{len(parts)}: {verdicts[-1].verdict}")
        if verdicts[-1].verdict != "continue":
            break
    name_a = f"A ({label_a}{'' if arm_a == 'main' else ', ' + arm_a})"
    name_b = f"B ({label_b}{'' if arm_b == 'main' else ', ' + arm_b})"
    psi = RP.discordant_share(rows_a, rows_b)
    text = "\n\n".join(
        [
            RP.summary(rows_a, name_a),
            RP.summary(rows_b, name_b),
            RP.paired(rows_a, rows_b, "A", "B"),
            RP.looks_table(verdicts, "A", "B"),
            f"Same-room clicks that ended differently: {100 * (psi or 0.0):.0f} %. Clicks a comparison like this one needs:",
            RP.power_table(sorted({round(psi or 0.25, 2), 0.15, 0.25, 0.35})),
        ]
    )
    return {
        "a": label_a,
        "b": label_b,
        "verdict": verdicts[-1].verdict if verdicts else None,
        "looks": verdicts,
        "text": text,
        "rows": (out_a, out_b),
    }


__all__ = ["compare", "run_tree", "same_room_look", "split_looks", "tree_of"]
