"""``python -m duckmind.bench gothere <command>`` (docs/validation/bench.md).

clicks    make a click set (seeded) or label an existing list of clicks
run       answer clicks with the tree this interpreter imports: headless, or ``--live URL`` on a running server
report    tables from one rows file, or two paired by click
compare   two source trees (git refs or snapshot folders) on the same clicks, with the stopping rule
power     how many clicks a wanted gain needs
roam      make a known-home map fixture (a seeded headless roam)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _ints(text: str | None) -> list[int] | None:
    """``"0,2-4"`` → [0, 2, 3, 4]."""
    if not text:
        return None
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _body_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--home", default="real_home")
    p.add_argument(
        "--policies",
        help="the walk policy folder (default: DUCKMIND_POLICY_DIR / the installed set)",
    )
    p.add_argument(
        "--collision",
        default=None,
        help="the robot's collision model (default: the live simulated duck's)",
    )


def _click_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--set",
        default="real_home-v1",
        dest="set_name",
        help="a committed set's name or a .jsonl path",
    )
    p.add_argument("--blocks", help="whole blocks of the set, e.g. 0-3")
    p.add_argument("--limit", type=int, default=0, help="the first n of the chosen clicks")
    p.add_argument("--keys", help="named clicks, e.g. 3,17-20")
    p.add_argument(
        "--clean",
        action="store_true",
        help="without the flagged clicks (a legacy set's unfair ones)",
    )


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="python -m duckmind.bench", description="The repository's measuring tools."
    )
    what = top.add_subparsers(dest="what", required=True)
    g = what.add_parser("gothere", help="the walk to a place somebody pointed at").add_subparsers(
        dest="cmd", required=True
    )

    p = g.add_parser("clicks", help="make or label a click set")
    p.add_argument("--home", default="real_home")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--name")
    p.add_argument(
        "--legacy",
        help="label this list of clicks (JSONL with pose / scene.truth and point) instead of making one",
    )
    p.add_argument("--out", help="the folder to write into (default: the bench's data folder)")

    p = g.add_parser("run", help="answer clicks")
    _click_args(p)
    _body_args(p)
    p.add_argument("--arm", default="main", help="main | truth | truth-reflex | truth-free")
    p.add_argument("--out", required=True, help="rows file (JSONL; rows already there are kept)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--known", help="the known-home fixture (a name, a path, 'none'; default: the home's own)"
    )
    p.add_argument("--trace", help="also write the 10 Hz traces here (.jsonl.gz)")
    p.add_argument("--tree-label", help="what to call this source tree in the rows")
    p.add_argument("--no-wait", action="store_true", help="do not wait for room on the machine")
    p.add_argument(
        "--live", help="the base URL of a running test server: drive it instead of the headless rig"
    )
    p.add_argument("--rec", help="(live) the server's --neuro-record file, for the 50 Hz counts")
    p.add_argument(
        "--live-clicks",
        type=int,
        default=0,
        help="(live) stop after this many clicks of the chosen ones",
    )

    p = g.add_parser("report", help="tables from rows")
    p.add_argument("rows", nargs="+", help="one rows file, or two to pair")
    p.add_argument("--names", help="A,B")

    p = g.add_parser("compare", help="two trees, the same clicks")
    p.add_argument("--a", required=True, help="the baseline: a git ref or a snapshot folder")
    p.add_argument("--b", required=True, help="the candidate")
    p.add_argument("--out", required=True, help="the comparison's folder (trees, rows, RESULT.md)")
    p.add_argument("--set", default="real_home-v1", dest="set_name")
    p.add_argument("--blocks", default="0-3")
    p.add_argument(
        "--looks",
        type=int,
        default=1,
        help="how many looks the stopping rule may take (1 = no early stop)",
    )
    p.add_argument("--delta", type=float, default=5.0, help="the gain worth having, in points")
    p.add_argument("--arm-a", default="main")
    p.add_argument("--arm-b", default="main")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-wait", action="store_true", help="do not wait for room on the machine")
    _body_args(p)

    p = g.add_parser("power", help="clicks needed")
    p.add_argument(
        "--discordant", default="15,25,35", help="shares of clicks that end differently, in percent"
    )
    p.add_argument("--p0", type=float, default=61.0)
    p.add_argument("--per-min", type=float, help="clicks a minute, to turn clicks into minutes")

    p = g.add_parser("roam", help="make a known-home fixture")
    _body_args(p)
    p.add_argument("--seed", type=int, default=9201)
    p.add_argument("--minutes", type=float, default=40.0)
    p.add_argument("--out", required=True)
    return top


def _collision(args: argparse.Namespace) -> str:
    from duckmind.bench import runner as RN

    return str(args.collision or RN.COLLISION)


def _policies(args: argparse.Namespace) -> str:
    from duckmind import paths

    return str(args.policies or paths.policy_dir())


def _chosen(args: argparse.Namespace) -> list[dict[str, Any]]:
    from duckmind.bench import clicks as C

    return C.select(
        C.load(args.set_name),
        blocks=_ints(args.blocks),
        limit=args.limit,
        keys=_ints(args.keys),
        clean=args.clean,
    )


def _cmd_clicks(args: argparse.Namespace) -> int:
    from duckmind.bench import clicks as C

    if args.legacy:
        rows = []
        for line in Path(args.legacy).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pose = (r.get("scene") or {}).get("truth") or r.get("pose")
            if r.get("ok", True) and pose:
                rows.append({"pose": pose, "point": r["point"]})
        name = args.name or "legacy"
        got, meta = C.relabel(
            args.home, rows, name, "live clicks of the earlier go-there measurements, replayed"
        )
    else:
        got, meta = C.generate(args.home, args.blocks, args.seed, args.name)
    out = C.save(got, meta, Path(args.out) if args.out else None)
    print(f"{len(got)} clicks at {out}")
    print(
        json.dumps({k: v for k, v in meta.items() if k in ("dropped", "flags")}, ensure_ascii=False)
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from duckmind.bench import report as RP

    chosen = _chosen(args)
    if args.live:
        from duckmind.bench import live as LV

        rows = LV.run_live(
            args.live,
            chosen,
            Path(args.out),
            home=args.home,
            rec=args.rec,
            tree=args.tree_label,
            max_clicks=args.live_clicks,
            log=_log,
        )
    else:
        from duckmind.bench import runner as RN

        rows = RN.run_batch(
            chosen,
            Path(args.out),
            arm=args.arm,
            workers=args.workers or None,
            known=args.known,
            home=args.home,
            collision=_collision(args),
            policies=_policies(args),
            tree=args.tree_label,
            trace=Path(args.trace) if args.trace else None,
            wait=not args.no_wait,
            log=_log,
        )
    print(RP.summary(rows, f"{args.arm}, {args.set_name}"))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from duckmind.bench import report as RP
    from duckmind.bench import runner as RN

    names = (args.names or "A,B").split(",")
    sets = [RN.load_rows(p) for p in args.rows[:2]]
    for rows, name, path in zip(sets, names, args.rows, strict=False):
        print(RP.summary(rows, f"{name}: {Path(path).name}"))
        print()
    if len(sets) == 2:
        print(RP.paired(sets[0], sets[1], names[0], names[1]))
        psi = RP.discordant_share(sets[0], sets[1])
        if psi is not None:
            print(f"\nSame-room clicks that ended differently: {100 * psi:.0f} %")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from duckmind.bench import compare as CP

    run_args = ["--home", args.home, "--collision", _collision(args), "--policies", _policies(args)]
    if args.workers:
        run_args += ["--workers", str(args.workers)]
    if args.no_wait:
        run_args.append("--no-wait")
    got = CP.compare(
        args.a,
        args.b,
        Path(args.out),
        set_name=args.set_name,
        blocks=_ints(args.blocks) or [0],
        looks=args.looks,
        delta=args.delta / 100.0,
        arm_a=args.arm_a,
        arm_b=args.arm_b,
        run_args=run_args,
        log=_log,
    )
    text = f"# {got['a']} vs {got['b']}: {got['verdict']}\n\n{got['text']}\n"
    (Path(args.out) / "RESULT.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    from duckmind.bench import report as RP

    shares = [float(x) / 100.0 for x in args.discordant.split(",")]
    print(RP.power_table(shares, args.p0 / 100.0, args.per_min))
    return 0


def _cmd_roam(args: argparse.Namespace) -> int:
    from duckmind.bench import knownhome as KH

    fixture = KH.roam(
        args.home,
        args.seed,
        args.minutes,
        collision=_collision(args),
        policies=_policies(args),
        log=_log,
    )
    Path(args.out).write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(fixture['map']['cells'])} cells, origin {fixture['origin']}, at {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return {
        "clicks": _cmd_clicks,
        "run": _cmd_run,
        "report": _cmd_report,
        "compare": _cmd_compare,
        "power": _cmd_power,
        "roam": _cmd_roam,
    }[args.cmd](args)


__all__ = ["main", "parser"]
