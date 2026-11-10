"""Tables from go-there rows (:data:`duckmind.bench.runner.SCHEMA`): one arm's summary, two arms paired by click, the
power table. Markdown, so a result file can take them as they are."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

from duckmind.bench import stats as S

Row = dict[str, Any]
LINES = ("clear", "low", "thing", "furniture", "wall")
GAINS = (0.05, 0.09, 0.15)


def groups(by_line: bool = True) -> list[tuple[str, Callable[[Row], bool]]]:
    """The classes a table is split by: all, the same room, another room, and the same room by what is on the line."""
    out: list[tuple[str, Callable[[Row], bool]]] = [
        ("all", lambda r: True),
        ("same room", lambda r: r.get("cls") == "same"),
        ("other room", lambda r: r.get("cls") == "other"),
    ]
    if by_line:
        for line in LINES:
            out.append(
                (
                    f"same room, line {line}",
                    lambda r, line=line: r.get("cls") == "same" and r.get("line") == line,
                )
            )
        out.append(
            ("same room, not flagged", lambda r: r.get("cls") == "same" and not r.get("flagged"))
        )
    return out


def pct(k: int, n: int) -> str:
    if n <= 0:
        return "–"
    lo, hi = S.wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f} % ({100 * lo:.0f}–{100 * hi:.0f})"


def summary(rows: Sequence[Row], name: str = "", by_line: bool = True) -> str:
    """One arm: arrivals within the time (Wilson 95 %), how long, and what went wrong, by class."""
    head = "| class | arrived ≤ 40 s (95 %) | … and truly within 25 cm | median s | falls (clicks) | bumps (clicks) | brushes | gave up | still walking | takeovers |"
    out = [f"**{name}**" if name else "", head, "|---|---|---|---|---|---|---|---|---|---|"]
    for label, keep in groups(by_line):
        rs = [r for r in rows if keep(r)]
        if not rs:
            continue
        arr = [r for r in rs if r.get("arrived")]
        secs = [float(r["arrived_s"]) for r in arr if r.get("arrived_s") is not None]
        truth = [r for r in rs if r.get("truth_ok")]
        known = [r for r in rs if r.get("truth_ok") is not None]
        med = f"{statistics.median(secs):.1f}" if secs else "–"
        out.append(
            f"| {label} | {pct(len(arr), len(rs))} | {pct(len(truth), len(known)) if known else '–'} | {med} "
            f"| {_total(rs, 'falls')} | {_total(rs, 'bumps')} | {sum(int(r.get('brushes') or 0) for r in rs)} "
            f"| {sum(1 for r in rs if r.get('gave_up'))} | {sum(1 for r in rs if r.get('state') == 'walking')} "
            f"| {sum(int(r.get('takeovers') or 0) for r in rs)} |"
        )
    return "\n".join(line for line in out if line)


def _total(rows: Sequence[Row], key: str) -> str:
    return f"{sum(int(r.get(key) or 0) for r in rows)} ({sum(1 for r in rows if r.get(key))})"


def pair(a: Sequence[Row], b: Sequence[Row]) -> list[tuple[Row, Row]]:
    """The clicks both arms answered, in click order."""
    bm = {(r.get("set"), int(r["k"])): r for r in b}
    return [(r, bm[(r.get("set"), int(r["k"]))]) for r in a if (r.get("set"), int(r["k"])) in bm]


def counts(pairs: Sequence[tuple[Row, Row]]) -> tuple[int, int, int, int]:
    """(both arrived, only A, only B, neither)."""
    both = sum(1 for x, y in pairs if x.get("arrived") and y.get("arrived"))
    only_a = sum(1 for x, y in pairs if x.get("arrived") and not y.get("arrived"))
    only_b = sum(1 for x, y in pairs if not x.get("arrived") and y.get("arrived"))
    return both, only_a, only_b, len(pairs) - both - only_a - only_b


def paired(
    a: Sequence[Row], b: Sequence[Row], name_a: str = "A", name_b: str = "B", by_line: bool = True
) -> str:
    """Two arms on the same clicks: arrivals by class with the exact McNemar test, then what went wrong, click by click."""
    pairs = pair(a, b)
    out = [
        f"| class | pairs | {name_a} | {name_b} | both | only {name_a} | only {name_b} | neither | {name_b} − {name_a} (95 %) | exact McNemar p |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, keep in groups(by_line):
        ps = [(x, y) for x, y in pairs if keep(x)]
        if not ps:
            continue
        both, oa, ob, neither = counts(ps)
        n = len(ps)
        d, lo, hi = S.paired_diff(oa, ob, n)
        out.append(
            f"| {label} | {n} | {pct(both + oa, n)} | {pct(both + ob, n)} | {both} | {oa} | {ob} | {neither} "
            f"| {100 * d:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f}) | {S.mcnemar_exact(oa, ob):.3f} |"
        )
    out += [
        "",
        f"| per click | class | {name_a} total (clicks) | {name_b} total (clicks) | clicks where {name_b} has more | fewer | exact sign-test p |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("falls", "bumps", "brushes", "gave_up", "takeovers"):
        for label, keep in groups(False):
            ps = [(x, y) for x, y in pairs if keep(x)]
            if not ps:
                continue
            more = sum(1 for x, y in ps if int(y.get(key) or 0) > int(x.get(key) or 0))
            fewer = sum(1 for x, y in ps if int(y.get(key) or 0) < int(x.get(key) or 0))
            out.append(
                f"| {key} | {label} | {_total([x for x, _ in ps], key)} | {_total([y for _, y in ps], key)} | {more} | {fewer} | {S.mcnemar_exact(fewer, more):.3f} |"
            )
    return "\n".join(out)


def discordant_share(a: Sequence[Row], b: Sequence[Row], cls: str | None = "same") -> float | None:
    ps = [(x, y) for x, y in pair(a, b) if cls is None or x.get("cls") == cls]
    if not ps:
        return None
    _, oa, ob, _ = counts(ps)
    return (oa + ob) / len(ps)


def power_table(
    shares: Sequence[float] = (0.15, 0.25, 0.35),
    p0: float = 0.61,
    per_min: float | None = None,
    same_share: float = 0.75,
) -> str:
    """Same-room clicks a paired comparison needs for a gain of 5 / 9 / 15 points (two-sided 5 %, power 80 %), by the
    share of clicks that end differently between the arms; the last row is two independent samples, for contrast."""
    head = (
        "| clicks that end differently | "
        + " | ".join(f"+{100 * g:.0f} points" for g in GAINS)
        + " |"
    )
    out = [head, "|---|" + "---|" * len(GAINS)]

    def cell(n: int) -> str:
        if per_min is None:
            return f"{n}"
        return f"{n} (≈ {2 * n / same_share / per_min:.0f} min for both arms)"

    for psi in shares:
        out.append(
            f"| {100 * psi:.0f} % (paired) | "
            + " | ".join(cell(S.paired_n(g, psi)) for g in GAINS)
            + " |"
        )
    out.append(
        f"| not paired, from {100 * p0:.0f} % | "
        + " | ".join(cell(S.unpaired_n(p0, g)) for g in GAINS)
        + " |"
    )
    return "\n".join(out)


def looks_table(looks: Sequence[S.Look], name_a: str = "A", name_b: str = "B") -> str:
    out = [
        f"| look | same-room pairs | only {name_a} | only {name_b} | {name_b} − {name_a} (repeated interval) | exact McNemar p | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, lk in enumerate(looks, 1):
        out.append(
            f"| {i} | {lk.n} | {lk.only_a} | {lk.only_b} | {100 * lk.diff:+.1f} ({100 * lk.lo:+.1f} to {100 * lk.hi:+.1f}) | {lk.p:.3f} | {lk.verdict} |"
        )
    return "\n".join(out)


__all__ = [
    "counts",
    "discordant_share",
    "groups",
    "looks_table",
    "pair",
    "paired",
    "pct",
    "power_table",
    "summary",
]
