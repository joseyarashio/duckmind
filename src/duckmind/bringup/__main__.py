"""``python -m duckmind.bringup <command>``: the bench tool of docs/deploy/bring-up.md.

  record URL --out DIR   record the duck over the link through the staged poses (appends to DIR's recording)
  fit DIR [--save]       fit the body calibration from DIR's recording, write the report; --save installs it
  calibrate URL --out DIR   record every stage, fit, and install the calibration when it passes
  show URL | --file F    which calibration this computer has for that body (or check one file)
  watch URL --minutes M  read only: M minutes of robot.state through proprio with that body's calibration — the
                         pickups, set-downs, falls and "unloaded" it would have claimed, the battery, the link

The recording is passive unless ``--drive`` (then the tool toggles sit/stand and commands the short bench walk itself).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from duckmind.bringup import fit as F
from duckmind.bringup import record as R
from duckmind.bringup import tofcal as T
from duckmind.duckbody import calibration as C

REPORT_MD, REPORT_JSON = "report.md", "fit.json"
UNITS = (
    "robot.state currents_ma, summed over the ten leg joints, as this body's robotd reports them"
)


def _link_arguments(parser: argparse.ArgumentParser) -> None:
    from duckmind import paths

    parser.add_argument("url", help="remote://HOST:PORT (the bridge on the duck, or a sim-body)")
    parser.add_argument(
        "--token-file", type=Path, default=paths.duckmind_home() / "remote-body.token"
    )


def _record_arguments(parser: argparse.ArgumentParser) -> None:
    _link_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="the session's folder")
    parser.add_argument(
        "--drive",
        action="store_true",
        help="the tool toggles sit/stand and commands the bench walk (default: it only reads, and "
        "the person does both with the duck's own control)",
    )
    parser.add_argument("--no-tof", action="store_true", help="skip the guided ToF measurement")
    parser.add_argument(
        "--unit", default="", help="your own label for this duck (a sticker, a name)"
    )
    parser.add_argument(
        "--sim-person",
        type=int,
        metavar="WORLD_PORT",
        help="a rehearsal: sim-body's world port; its simulated hand plays the person",
    )


def header_for(info: Mapping[str, Any], *, tof: bool, unit: str) -> dict[str, Any]:
    return {
        "started": R.now_iso(),
        "body": dict(info),
        "key": list(C.key_of(info)),
        "tof": tof,
        "unit": unit,
    }


def record(args: argparse.Namespace, stages: Sequence[int]) -> int:
    from duckmind.bringup.link import LinkBody
    from duckmind.remotebody.link import read_token

    body = LinkBody(args.url, read_token(args.token_file))
    person: Any = None
    try:
        key = C.key_of(body.info)
        print(f"link up: board {key[0]}, walk {key[1]}, sit-stand {key[2]}")
        if None in key:
            print(
                "the bridge did not say the board or the networks: a calibration could not be "
                "matched to this body (is this a real duck's bridge that can read robotd.toml?)",
                file=sys.stderr,
            )
            return 2
        if body.info.get("backend") != "real":
            print("this body is a plain simulated one: it senses with the simulator's numbers")
        path = args.out / R.RECORDING
        if path.exists():
            before, _ = R.read(path)
            if before.get("key") != list(key):
                print(
                    f"{path} was recorded on another body or with other networks ({before.get('key')}): "
                    "use a new folder",
                    file=sys.stderr,
                )
                return 2
        tof = body.has_tof and not args.no_tof
        if args.sim_person is not None:
            from duckmind.bringup.simperson import SimPerson

            person = SimPerson(args.sim_person)
        operator = person if person is not None else R.Console()
        sink = R.Sink(path)
        sink.open(header_for(body.info, tof=body.has_tof, unit=args.unit))
        try:
            done = R.record(body, R.plan_for(stages, tof=tof), sink, operator, drive=args.drive)
        finally:
            sink.close()
        for name, ticks in done.items():
            print(f"  {name}: {ticks} ticks")
        print(f"recorded to {path}")
        return 0
    except (R.SegmentFailed, KeyboardInterrupt) as error:
        print(f"stopped: {error or 'interrupted'} (the duck was told to stop)", file=sys.stderr)
        return 1
    finally:
        if person is not None:
            person.close()
        body.close()


def build(
    header: Mapping[str, Any], legs: F.Fit, tof: T.TofFit | None, digest: str
) -> C.Calibration | None:
    """The calibration document for a fit that passed (None: it did not, and :func:`why_not` says why)."""
    if why_not(header, legs, tof):
        return None
    board, walk, seat = header["key"]
    bands = (
        C.SIMULATOR.tof
        if board == C.REHEARSAL_BOARD and (tof is None or tof.bands is None)
        else None
    )
    if tof is not None and tof.bands is not None:
        bands = tof.bands
    assert legs.bands is not None
    return C.Calibration(
        board=board,
        walk_sha256=walk,
        sitstand_sha256=seat,
        source="bench",
        proprio=legs.bands,
        tof=bands,
        recorded=R.now_iso(),
        unit=str(header.get("unit") or ""),
        units=UNITS,
        evidence={
            "recording_sha256": digest,
            "lifts": legs.replay.get("lifts"),
            "claim_after_cue_s": legs.replay.get("claim_after_cue_s"),
            "cannot_separate": list(legs.cannot),
            "tof": "none fitted"
            if bands is None
            else (
                "measured"
                if tof is not None and tof.bands is not None
                else "rehearsal: the simulator's sensor"
            ),
        },
    )


def why_not(header: Mapping[str, Any], legs: F.Fit, tof: T.TofFit | None) -> list[str]:
    """Why this recording gives no calibration yet (empty: it does)."""
    why = list(legs.missing)
    if legs.bands is None and not why:
        why.append("no leg-load bands could be fitted")
    key = header.get("key") or [None, None, None]
    if None in key:
        why.append("the recording does not say the board and both networks")
    rehearsal = key[0] == C.REHEARSAL_BOARD
    if header.get("tof") and not rehearsal:
        if tof is None:
            why.append("this duck has a ToF and the guided ToF measurement was not recorded")
        elif tof.bands is None:
            why.append(f"the ToF touch band: {tof.why}")
    return why


def _table(rows: Sequence[Sequence[Any]]) -> list[str]:
    head, *body = rows
    out = ["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)]
    return out + [
        "| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in body
    ]


def report(header: Mapping[str, Any], legs: F.Fit, tof: T.TofFit | None, why: Sequence[str]) -> str:
    """The bench report (Markdown): the distributions, the separations, the bands, the replay, what cannot be told
    apart, and the verdict."""
    board, walk, seat = (header.get("key") or [None] * 3)[:3]
    lines = [
        "# Body calibration report",
        "",
        f"- board `{board}`, unit `{header.get('unit') or '-'}`, recorded from {header.get('started')}",
        f"- walk `{walk}`",
        f"- sit-stand `{seat}`",
        f"- **verdict: {'PASS — the calibration can be installed' if not why else 'NOT YET'}**",
    ]
    lines += [f"  - {reason}" for reason in why]
    lines += ["", "## Leg load by pose (this body's own units)", ""]
    keys = ("n", "p01", "p05", "p50", "p95", "p99", "min", "max")
    lines += _table(
        [("pose", *keys)]
        + [(name, *(level.get(k) for k in keys)) for name, level in legs.levels.items()]
    )
    lines += ["", f"## Separations (needs a ratio of {F.MIN_RATIO} or more)", ""]
    lines += _table(
        [("claim", "low level", "high level", "ratio", "separated")]
        + [
            (
                s.name + ("" if s.needed else " (optional)"),
                f"{s.low_name} {s.low:.1f}",
                f"{s.high_name} {s.high:.1f}",
                f"{s.ratio:.2f}",
                "yes" if s.ok else "NO",
            )
            for s in legs.separations
        ]
    )
    if legs.bands is not None:
        mine, sim = legs.bands.to_json(), C.SIMULATOR.proprio.to_json()
        lines += ["", "## Fitted bands", ""]
        lines += _table(
            [("band", "this body", "the simulator's")] + [(k, mine[k], sim[k]) for k in mine]
        )
    if legs.gyro:
        lines += ["", "## The quiet-trunk test (checked, not fitted)", ""]
        lines += [f"- {k}: {v}" for k, v in legs.gyro.items()]
    lines += ["", "## What this body's leg load cannot tell apart", ""]
    lines += [f"- {text}" for text in legs.cannot] or ["- nothing that a claim needs"]
    if legs.replay:
        lines += ["", "## Replay through proprio with the fitted bands", ""]
        lifts = legs.replay["lifts"]
        lines.append(
            "- pickups claimed: "
            + ", ".join(f"{name} {v['claimed']}/{v['of']}" for name, v in lifts.items())
            + f"; claim after the LIFT cue: {legs.replay['claim_after_cue_s']} (the person's reaction included)"
        )
        lines += [f"- FAIL {text}" for text in legs.replay["failures"]] or [
            "- no false claim, no missed lift"
        ]
        lines.append(f"- sit-stand cycles recorded without a fall: {legs.replay['cycles']}")
        lines += [f"- note: {text}" for text in legs.replay["notes"]]
        lines += [""]
        lines += _table(
            [("segment", "rep", "s", "edges (s, name)")]
            + [
                (
                    r["segment"],
                    r["rep"] + 1,
                    r["seconds"],
                    " ".join(f"{t}:{e}" for t, e in r["edges"]) or "-",
                )
                for r in legs.replay["rows"]
            ]
        )
    if tof is not None:
        data = tof.to_json()
        lines += ["", "## ToF touch band", ""]
        if tof.bands is not None:
            kept = " (the simulator's numbers fit this sensor)" if tof.kept_simulator else ""
            lines.append(
                f"- too-close band {tof.bands.too_close_mm} mm, touch at {tof.bands.touch_zones} zones{kept}; "
                f"gap {tof.gap} zones"
            )
        else:
            lines.append(f"- NOT FITTED: {tof.why}")
        lines += [""]
        lines += _table(
            [
                (
                    "board mm",
                    "frames",
                    "valid zones p50",
                    "nearest return mm",
                    "too-close zones min/p50/max",
                )
            ]
            + [
                (
                    row["board_mm"],
                    row["frames"],
                    row["valid_zones_p50"],
                    row["nearest_return_mm_p50"],
                    "{min}/{p50}/{max}".format(**row["too_close_zones"]),
                )
                for row in data["rows"]
            ]
        )
        lines.append(f"\n(counts shown for the {data['shown_band_mm']} mm band)")
    return "\n".join(lines) + "\n"


def fit(folder: Path, *, save: bool, where: Path | None = None) -> int:
    path = folder / R.RECORDING
    header, lines = R.read(path)
    legs, tof = F.fit(lines), T.fit(lines)
    why = why_not(header, legs, tof)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calibration = build(header, legs, tof, digest)
    (folder / REPORT_MD).write_text(report(header, legs, tof, why), encoding="utf-8")
    (folder / REPORT_JSON).write_text(
        json.dumps(
            {
                "key": header.get("key"),
                "recording_sha256": digest,
                "why_not": why,
                "legs": legs.to_json(),
                "tof": None if tof is None else tof.to_json(),
                "calibration": None if calibration is None else calibration.to_json(),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report: {folder / REPORT_MD}")
    if calibration is None:
        print("no calibration yet:")
        for reason in why:
            print(f"  - {reason}")
        return 1
    print(f"fitted: {json.dumps(calibration.proprio.to_json())}")
    print(f"ToF: {None if calibration.tof is None else calibration.tof.to_json()}")
    if save:
        print(f"installed: {C.save(calibration, where)} (the brain reads it at its next start)")
    else:
        print("passes; run again with --save to install it for the brain")
    return 0


def show(args: argparse.Namespace) -> int:
    if args.file is not None:
        try:
            cal = C.load(args.file)
        except C.CalibrationError as error:
            print(f"{args.file}: not usable: {error}")
            return 1
        print(json.dumps(cal.to_json(), indent=1))
        return 0
    from duckmind.remotebody.client import RemoteLink
    from duckmind.remotebody.link import read_token

    link = RemoteLink(args.url, read_token(args.token_file))
    try:
        link.start(timeout_s=10)
        key = C.key_of(link.info)
    finally:
        link.close()
    found = C.find(*key)
    print(f"body: board {key[0]}, walk {key[1]}, sit-stand {key[2]}")
    if found.calibration is None:
        print(
            f"no valid calibration ({found.why}{': ' + found.detail if found.detail else ''}): head only"
        )
        return 1
    print(f"calibration: {found.path} (recorded {found.calibration.recorded})")
    return 0


def watch(args: argparse.Namespace) -> int:
    """Hold the link ``--minutes`` and send nothing: every tick goes through proprio with the body's installed
    calibration (the simulator's numbers for a body that has none: then the claims only show what is missing), the
    battery is read every 30 s and the link's own numbers at the end. Writes ``watch-<time>.json`` into ``--out``;
    exit 0 when nothing was claimed and the link never dropped."""
    import time

    from duckmind.bringup.link import LinkBody
    from duckmind.duckbody.proprio import Proprio, ProprioConfig
    from duckmind.duckbody.robotd_client import RobotdError, RobotState
    from duckmind.remotebody.battery import parse
    from duckmind.remotebody.link import read_token

    body = LinkBody(args.url, read_token(args.token_file))
    try:
        key = C.key_of(body.info)
        found = C.find(*key)
        cfg = ProprioConfig()
        if found.calibration is not None:
            cfg = cfg.calibrated(found.calibration.proprio)
        print(
            f"watching board {key[0]} for {args.minutes} min with {found.path or 'NO calibration'}"
        )
        proprio = Proprio(cfg)
        edges: dict[str, int] = {}
        policies: dict[str, int] = {}
        battery: list[dict[str, Any]] = []
        loads: list[float] = []
        ticks, first, last, asked = 0, None, 0, -1e9
        span, began = args.minutes * 60.0, time.monotonic()
        # (the duck's own clock; this computer's as the backstop for a robotd that sends none)
        while (
            first is None or (last - first) / 1e9 < span
        ) and time.monotonic() - began < span + 5.0:
            params = body.read_state(2.0)
            if params is None:
                print("robot.state stopped coming", file=sys.stderr)
                break
            state = RobotState.from_wire(params)
            first, last, ticks = state.t_ns if first is None else first, state.t_ns, ticks + 1
            policies[state.policy] = policies.get(state.policy, 0) + 1
            if state.leg_load is not None:
                loads.append(state.leg_load)
            for edge in proprio.step(state).edges:
                edges[edge] = edges.get(edge, 0) + 1
                print(f"  {(last - first) / 1e9:7.1f} s  {edge}")
            if time.monotonic() - asked >= 30.0:
                asked = time.monotonic()
                try:
                    reading = parse(body.health())
                except RobotdError:  # (a robotd whose health has no verdict: no battery reading)
                    reading = None
                if reading is not None:
                    battery.append({"s": round((last - first) / 1e9, 1), **reading})
        stats = body.link.stats()
    finally:
        body.close()
    percents = [b["percent"] for b in battery]
    result = {
        "key": list(key),
        "calibration": None if found.path is None else str(found.path.name),
        "seconds": 0.0 if first is None else round((last - first) / 1e9, 1),
        "ticks": ticks,
        "policies": policies,
        "edges": edges,
        "leg_load": F.describe(loads),
        "currents_reported": bool(loads),
        "battery_percent": None
        if not percents
        else {"first": percents[0], "min": min(percents), "last": percents[-1]},
        "battery": battery,
        "link": {k: stats.get(k) for k in ("drops", "max_gap_ms", "rtt_ms_p50", "rtt_ms_p95")},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"watch-{R.now_iso().replace(':', '')}.json"
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "battery"}))
    print(f"written: {path}")
    claimed = sum(edges.get(name, 0) for name in ("lifted", "unloaded", "fell"))
    return 0 if ticks and not claimed and not stats.get("drops") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckmind.bringup", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="record the staged poses over the link")
    _record_arguments(rec)
    rec.add_argument(
        "--stages",
        default="1,2,3",
        help="which stages to record (1: in the hand, 2: floor, 3: walk)",
    )
    fitp = commands.add_parser("fit", help="fit the calibration from a session's recording")
    fitp.add_argument("folder", type=Path)
    fitp.add_argument("--save", action="store_true", help="install the calibration when it passes")
    fitp.add_argument(
        "--where", type=Path, help="install into this folder (default: the duckmind home's)"
    )
    cal = commands.add_parser("calibrate", help="record every stage, fit, install")
    _record_arguments(cal)
    cal.add_argument(
        "--where", type=Path, help="install into this folder (default: the duckmind home's)"
    )
    shown = commands.add_parser("show", help="which calibration applies to a body")
    shown.add_argument("url", nargs="?")
    shown.add_argument("--file", type=Path)
    from duckmind import paths

    shown.add_argument(
        "--token-file", type=Path, default=paths.duckmind_home() / "remote-body.token"
    )
    watched = commands.add_parser(
        "watch", help="read only: what proprio would claim, the battery, the link"
    )
    _link_arguments(watched)
    watched.add_argument("--out", type=Path, required=True)
    watched.add_argument("--minutes", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.command == "watch":
        return watch(args)
    if args.command == "record":
        return record(args, [int(s) for s in str(args.stages).split(",") if s.strip()])
    if args.command == "fit":
        return fit(args.folder, save=args.save, where=args.where)
    if args.command == "calibrate":
        code = record(args, (1, 2, 3))
        return code if code else fit(args.out, save=True, where=args.where)
    if args.file is None and args.url is None:
        parser.error("show needs a URL or --file")
    return show(args)


if __name__ == "__main__":
    sys.exit(main())
