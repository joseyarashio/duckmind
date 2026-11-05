"""Rehearse the remote body: the same brain-side body code against a local and a remote body.

``local``  — the simulator page's own body (robotd + MuJoCo started in this process's children);
``remote`` — ``python -m duckmind.remotebody sim-body`` as a second process, reached through the
             link and a :class:`~duckmind.remotebody.netshim.NetShim` (delay, jitter, loss) in this
             process; ``--cuts N`` adds N runs that cut the link mid-walk and restore it.

Each run, on the brain's side (a :class:`~duckmind.livebody.LiveBody`, exactly what the page uses):
the closed-loop command latency (``robot.head`` sent → robotd's state frame showing it, back on the
brain's side; 20 probes), stand up, walk 3 s at 0.15 m/s, turn for 2 s, kick (``robot.do
kick_left``), sit. A cut run walks, cuts the link, and times the duck's stop and sit; then restores
the link and times the brain's return and a fresh walk. The scorer reads the world's truth
(trunk position, robotd's policy) over its own connections, never the brain's.

    python tools/measure_remote_body.py local  --runs 10 --out OUT  <live paths>
    python tools/measure_remote_body.py remote --runs 10 --cuts 10 --token-file T --out OUT <live paths>

Live paths: ``--microduck --microduck-rl --policies --robotd-bin`` (as ``python -m duckmind.simweb``).
Results: ``OUT/<mode>-runs.jsonl``, ``OUT/<mode>-cuts.jsonl`` and a Markdown table on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from duckmind import bodyhost
from duckmind.body import BodyIntent
from duckmind.livebody import LiveBody, LivePaths, RemoteBodySpec
from duckmind.remotebody.netshim import Impairment, NetShim
from duckmind.robotd import RobotdBodyPort
from duckmind.sim_live import SimOracle, _installed_ort_library

# The walking network's slowest asked-for walk (Duck Core's walk_vx floor; slower asks step in
# place: about 0.1 m/s covered from a standstill) and a turn on the move (it does not turn on
# the spot: an in-place vyaw 1.0 turned 0.07 rad in 3 s).
WALK_VX, WALK_S = 0.30, 3.0
TURN_VYAW, TURN_S = 1.0, 2.0
SEATED_Z = 0.085  # trunk height (m): ~0.06 seated, ~0.12 standing
WALK_OK_M = 0.15
TURN_OK_RAD = 0.5
STOPPED_MPS = 0.03
STOPPED_FOR_S = 0.3
PROBES = 20
UPRIGHT = ("walk", "stand")


def yaw_of(quat: list[float]) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Scorer:
    """Truth for scoring only: the trunk from the world's port, the policy from robotd's own
    socket, both on this machine and never through the brain's link."""

    def __init__(self, world_port: int, robotd: str) -> None:
        self.oracle = SimOracle(world_port)
        self.world = bodyhost.HostClient(world_port, timeout_s=5.0)
        self.port = RobotdBodyPort(robotd, backend="simulator", hz=50)
        self.samples: list[tuple[float, ...]] = []  # wall, sim, x, y, yaw, z
        self.states: list[tuple[float, str, bool]] = []  # wall, policy, fallen
        self._stop = threading.Event()
        for target in (self._truth, self._policy):
            threading.Thread(target=target, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        for closer in (self.oracle, self.world, self.port):
            try:
                closer.close()
            except Exception:
                pass

    def _truth(self) -> None:
        while not self._stop.is_set():
            try:
                if self.oracle.lost:
                    self.oracle.reconnect()
                snap = self.oracle.snapshot()
            except Exception:
                time.sleep(0.05)
                continue
            trunk = snap.get("trunk") or [0.0, 0.0, 0.0]
            quat = (snap.get("imu") or {}).get("quat") or [1.0, 0.0, 0.0, 0.0]
            self.samples.append(
                (
                    time.monotonic(),
                    float(snap.get("sim_time", 0.0)),
                    trunk[0],
                    trunk[1],
                    yaw_of(quat),
                    trunk[2],
                )
            )
            time.sleep(0.02)

    def _policy(self) -> None:
        while not self._stop.is_set():
            try:
                seen = self.port.observe()
            except Exception:
                time.sleep(0.05)
                continue
            self.states.append((time.monotonic(), seen.policy, bool(seen.safety.fallen)))

    @property
    def policy(self) -> str:
        return self.states[-1][1] if self.states else "?"

    def pose(self) -> tuple[float, float, float]:
        _, _, x, y, yaw, _ = self.samples[-1]
        return x, y, yaw

    def speed_since(self, since: float) -> list[tuple[float, float]]:
        """(wall, planar speed m/s over simulated time) for samples after ``since``."""
        out = []
        rows = [s for s in self.samples if s[0] >= since - 0.1]
        for a, b in zip(rows, rows[1:]):
            dt = b[1] - a[1]
            if dt > 1e-4:
                out.append((b[0], math.hypot(b[2] - a[2], b[3] - a[3]) / dt))
        return out

    def timeline(self, since: float) -> list[tuple[float, str, bool]]:
        """robotd's policy (and fallen) each time it changed after ``since`` (s from then)."""
        out: list[tuple[float, str, bool]] = []
        for t, policy, fallen in list(self.states):
            if t >= since and (not out or out[-1][1:] != (policy, fallen)):
                out.append((round(t - since, 2), policy, fallen))
        return out

    def falls(self, since: float) -> int:
        rows = [f for t, _, f in self.states if t >= since]
        return sum(1 for a, b in zip(rows, rows[1:]) if b and not a)

    def place(self, x: float, y: float, yaw: float) -> bool:
        self.world.send(
            "x_place",
            x_mm=round(x * 1000),
            y_mm=round(y * 1000),
            yaw_mrad=round(yaw * 1000),
            keyframe="SIT",
        )
        return bool(self.world.receive("x_place").get("ok"))


def wait(check: Any, timeout_s: float, step: float = 0.02) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if check():
            return time.monotonic() - start
        time.sleep(step)
    return None


def latency_probe(body: LiveBody, robotd: Path) -> list[float]:
    """Closed loop: robot.head out through the body's channel, robotd's state frame showing it
    back on the brain's side (ms)."""
    body.passive = True  # (the body's own 50 Hz head stream would overwrite the probe)
    time.sleep(0.1)
    out: list[float] = []
    port = RobotdBodyPort(robotd, backend="simulator", hz=50)
    try:
        for k in range(PROBES):
            value = round(0.004 * (1 + k % 5) * (1 if k % 2 else -1), 4)
            sent = time.monotonic()
            port.dispatch_intent(
                BodyIntent(
                    "robot.head",
                    {"neck_pitch": value, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0},
                )
            )
            while time.monotonic() - sent < 2.0:
                frame = port._state.wire.read(2.0)
                head = (frame.get("params") or {}).get("head") or [None]
                if head[0] is not None and abs(float(head[0]) - value) < 1e-6:
                    out.append((time.monotonic() - sent) * 1000.0)
                    break
            time.sleep(0.05)
    finally:
        port.close()
        body.passive = False
    return out


def posture(body: LiveBody, scorer: Scorer, want: str, timeout_s: float = 12.0) -> float | None:
    body.request_posture(want)
    ok = (lambda: scorer.policy in UPRIGHT) if want == "stand" else (lambda: scorer.policy == "sit")
    return wait(ok, timeout_s)


def one_run(
    body: LiveBody, scorer: Scorer, start: tuple[float, float, float], robotd: Path
) -> dict[str, Any]:
    began = time.monotonic()
    drops = body._link.drops if body._link is not None else 0
    row: dict[str, Any] = {}
    if scorer.policy != "sit":
        posture(body, scorer, "sit")
    row["placed"] = scorer.place(*start)
    time.sleep(1.0)
    probes = latency_probe(body, robotd)
    row["latency_ms"] = probes
    row["stand_s"] = posture(body, scorer, "stand")
    time.sleep(0.5)
    x0, y0, yaw0 = scorer.pose()
    body.consume([{"walk": {"vx": WALK_VX, "vyaw": 0.0, "seconds": WALK_S}}])
    wait(lambda: body.walking, 2.0)
    wait(lambda: not body.walking, WALK_S + 8.0)
    time.sleep(0.6)
    x1, y1, yaw1 = scorer.pose()
    row["walk_m"] = round(math.hypot(x1 - x0, y1 - y0), 3)
    if scorer.policy not in UPRIGHT:  # (a brain knows its posture: a lost link sits it down)
        posture(body, scorer, "stand")
    body.consume([{"walk": {"vx": WALK_VX, "vyaw": TURN_VYAW, "seconds": TURN_S}}])
    wait(lambda: body.walking, 2.0)
    wait(lambda: not body.walking, TURN_S + 8.0)
    time.sleep(0.6)
    _, _, yaw2 = scorer.pose()
    row["turn_rad"] = round(abs(math.remainder(yaw2 - yaw1, math.tau)), 3)
    if scorer.policy not in UPRIGHT:
        posture(body, scorer, "stand")
    kicked = time.monotonic()
    body.skill("kick_left")
    seen = wait(lambda: any(t >= kicked and p not in UPRIGHT for t, p, _ in scorer.states), 4.0)
    back = wait(lambda: scorer.policy in UPRIGHT, 10.0) if seen is not None else None
    row["kick"] = {"ack": body.skill_ack, "started_s": seen, "back_s": back}
    row["sit_s"] = posture(body, scorer, "sit")
    row["falls"] = scorer.falls(began)
    row["policies"] = scorer.timeline(began)
    row["link_drops"] = (body._link.drops - drops) if body._link is not None else 0
    row["realtime"] = body.realtime_factor
    row["walk_ok"] = row["walk_m"] >= WALK_OK_M and row["falls"] == 0
    row["turn_ok"] = row["turn_rad"] >= TURN_OK_RAD and row["falls"] == 0
    row["kick_ok"] = (
        bool((body.skill_ack or {}).get("accepted") and seen is not None and back is not None)
        and row["falls"] == 0
    )
    return row


def events_since(path: Path | None, wall: float) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("t", 0) >= wall:
            out.append(event)
    return out


def cut_run(
    body: LiveBody,
    scorer: Scorer,
    shim: NetShim,
    start: tuple[float, float, float],
    events: Path | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if scorer.policy != "sit":
        posture(body, scorer, "sit")
    scorer.place(*start)
    time.sleep(1.0)
    row["stand_s"] = posture(body, scorer, "stand")
    time.sleep(0.5)
    began = time.monotonic()
    body.consume([{"walk": {"vx": WALK_VX, "vyaw": 0.0, "seconds": 8.0}}])
    moving = wait(lambda: any(v > 0.05 for _, v in scorer.speed_since(time.monotonic() - 0.3)), 6.0)
    time.sleep(1.0)
    row["moving_before_cut"] = moving is not None
    wall_cut = time.time()
    cut = shim.cut()
    stopped_at = None
    sat_at = None
    deadline = cut + 10.0
    while time.monotonic() < deadline and (stopped_at is None or sat_at is None):
        speeds = scorer.speed_since(cut)
        if stopped_at is None:
            for i, (t, v) in enumerate(speeds):
                if (
                    v < STOPPED_MPS
                    and all(w < STOPPED_MPS for t2, w in speeds[i:] if t2 - t <= STOPPED_FOR_S)
                    and speeds[-1][0] - t >= STOPPED_FOR_S
                ):
                    stopped_at = t
                    break
        if sat_at is None:
            sat_at = next((s[0] for s in scorer.samples if s[0] >= cut and s[5] < SEATED_Z), None)
        time.sleep(0.05)
    row["stop_s"] = None if stopped_at is None else round(stopped_at - cut, 3)
    row["sit_s"] = None if sat_at is None else round(sat_at - cut, 3)
    seen = events_since(events, wall_cut)
    first = {e["event"]: e["t"] - wall_cut for e in reversed(seen)}
    row["bridge_lost_s"] = round(first["link_lost"], 3) if "link_lost" in first else None
    row["bridge_stop_s"] = round(first["stop_sent"], 3) if "stop_sent" in first else None
    row["bridge_sit_s"] = round(first["sit_sent"], 3) if "sit_sent" in first else None
    link = body._link
    row["brain_lost_s"] = (
        round(link.down_since - cut, 3) if link is not None and link.down_since else None
    )
    restored = shim.restore()
    up = wait(lambda: link.up, 15.0)
    resumed = wait(lambda: body.state_at > restored, 15.0)
    row["link_back_s"] = None if up is None else round(up, 3)
    row["brain_back_s"] = None if resumed is None else round(time.monotonic() - restored, 3)
    # the brain drives again: stand, walk 2 s
    row["restand_s"] = posture(body, scorer, "stand")
    x0, y0, _ = scorer.pose()
    body.consume([{"walk": {"vx": WALK_VX, "vyaw": 0.0, "seconds": 2.0}}])
    wait(lambda: body.walking, 2.0)
    wait(lambda: not body.walking, 10.0)
    time.sleep(0.6)
    x1, y1, _ = scorer.pose()
    row["walk_after_m"] = round(math.hypot(x1 - x0, y1 - y0), 3)
    row["falls"] = scorer.falls(began)
    row["realtime"] = body.realtime_factor
    return row


def summarise(mode: str, runs: list[dict[str, Any]], cuts: list[dict[str, Any]]) -> str:
    def med(values: list[float]) -> str:
        return f"{statistics.median(values):.0f}" if values else "–"

    def p95(values: list[float]) -> str:
        return f"{sorted(values)[int(0.95 * (len(values) - 1))]:.0f}" if values else "–"

    lat = [v for r in runs for v in r["latency_ms"]]
    lines = [
        f"### {mode}: {len(runs)} runs",
        "",
        "| command latency p50 / p95 ms | walk ok | turn ok | kick ok | falls | link drops "
        "| realtime p50 |",
        "|---|---|---|---|---|---|---|",
        f"| {med(lat)} / {p95(lat)} | {sum(r['walk_ok'] for r in runs)}/{len(runs)} | "
        f"{sum(r['turn_ok'] for r in runs)}/{len(runs)} | {sum(r['kick_ok'] for r in runs)}/{len(runs)} "
        f"| {sum(r['falls'] for r in runs)} | {sum(r.get('link_drops', 0) for r in runs)} "
        f"| {statistics.median(r['realtime'] for r in runs):.2f} |",
    ]
    if cuts:

        def col(key: str) -> list[float]:
            return [float(c[key]) for c in cuts if c.get(key) is not None]

        lines += [
            "",
            f"### {mode}: {len(cuts)} link cuts mid-walk",
            "",
            "| | median s | max s | n |",
            "|---|---|---|---|",
        ]
        for key, label in (
            ("bridge_lost_s", "bridge declares the link lost"),
            ("bridge_stop_s", "bridge sends robot.stop"),
            ("stop_s", "duck stopped (truth < 0.03 m/s for 0.3 s)"),
            ("bridge_sit_s", "bridge sends sit"),
            ("sit_s", f"duck seated (truth: trunk below {SEATED_Z * 100:.1f} cm)"),
            ("brain_lost_s", "brain notices the link is down"),
            ("link_back_s", "link back after restore"),
            ("brain_back_s", "brain's body gets state again"),
        ):
            values = col(key)
            lines.append(
                f"| {label} | {statistics.median(values):.2f} | {max(values):.2f} | {len(values)} |"
                if values
                else f"| {label} | – | – | 0 |"
            )
        walked = sum(1 for c in cuts if (c.get("walk_after_m") or 0) >= WALK_OK_M / 2)
        lines.append(
            f"\nwalks again after the restore: {walked}/{len(cuts)}; "
            f"falls in cut runs: {sum(c['falls'] for c in cuts)}"
        )
    return "\n".join(lines)


def paths_of(args: argparse.Namespace) -> LivePaths:
    ort = args.ort_lib or _installed_ort_library()
    return LivePaths(args.microduck, args.microduck_rl, args.policies, args.robotd_bin, ort)


def launch_body(
    args: argparse.Namespace, events: Path
) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    command = [
        sys.executable, "-m", "duckmind.remotebody", "sim-body",
        "--token-file", str(args.token_file), "--listen", f"127.0.0.1:{args.body_port}",
        "--events", str(events), "--microduck", str(args.microduck),
        "--microduck-rl", str(args.microduck_rl), "--policies", str(args.policies),
        "--robotd-bin", str(args.robotd_bin), "--plain",
    ]  # fmt: skip
    if args.home:
        command += ["--home", args.home]
    if args.no_nest:
        command.append("--no-nest")
    log = open(args.out / "simbody.log", "w")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 600
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        if line.startswith("remote-body: ready "):
            ready = json.loads(line.split("ready ", 1)[1])

            def drain() -> None:
                for rest in process.stdout:  # type: ignore[union-attr]
                    log.write(rest)
                    log.flush()

            threading.Thread(target=drain, daemon=True).start()
            return process, ready
        if time.monotonic() > deadline:
            break
    process.kill()
    raise RuntimeError("the simulated body never became ready (see simbody.log)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("local", "remote"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cuts", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--home")
    parser.add_argument("--no-nest", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--body-port", type=int, default=8797)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--jitter-ms", type=float, default=15.0)
    parser.add_argument("--loss", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--soak-s", type=float, default=0.0, help="remote: then hold the link this long, seated"
    )
    for name in ("--microduck", "--microduck-rl", "--policies", "--robotd-bin", "--ort-lib"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DUCKMIND_ROBOTD_HEAD_ALPHA", "1.0")
    paths = paths_of(args)
    process = None
    shim = None
    events: Path | None = None
    if args.mode == "local":
        body = LiveBody(paths, seated=True, home=args.home, nest=False if args.no_nest else None)
        body.start()
        world_port, robotd = int(body.body_port or 0), str(body.robotd_socket)
        channel = Path(robotd)
    else:
        if args.token_file is None:
            parser.error("remote needs --token-file")
        events = args.out / "bridge-events.jsonl"
        process, ready = launch_body(args, events)
        world_port, robotd = int(ready["world_port"]), str(ready["robotd"])
        shim = NetShim(
            ("127.0.0.1", int(ready["port"])),
            impairment=Impairment(args.delay_ms, args.jitter_ms, args.loss),
            seed=args.seed,
        ).start()
        body = LiveBody(
            paths,
            seated=True,
            remote=RemoteBodySpec(f"remote://127.0.0.1:{shim.port}", args.token_file),
        )
        body.start()
        channel = body._link.robotd_socket
    scorer = Scorer(world_port, robotd)
    runs: list[dict[str, Any]] = []
    cuts: list[dict[str, Any]] = []
    try:
        wait(lambda: len(scorer.samples) > 5 and scorer.states, 10.0)
        start = scorer.pose()
        with open(args.out / f"{args.mode}-runs.jsonl", "w") as out:
            for index in range(args.runs):
                row = one_run(body, scorer, start, channel)
                row["run"] = index
                runs.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"run {index}: {json.dumps(row)}", flush=True)
        if shim is not None and args.cuts:
            with open(args.out / f"{args.mode}-cuts.jsonl", "w") as out:
                for index in range(args.cuts):
                    row = cut_run(body, scorer, shim, start, events)
                    row["run"] = index
                    cuts.append(row)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(f"cut {index}: {json.dumps(row)}", flush=True)
        if shim is not None and args.soak_s > 0:
            before = body._link.drops
            gaps_before = body._link.max_gap_ms
            time.sleep(args.soak_s)
            soak = {
                "seconds": args.soak_s,
                "delay_ms": args.delay_ms,
                "jitter_ms": args.jitter_ms,
                "loss": args.loss,
                "link_drops": body._link.drops - before,
                "max_gap_ms_before": round(gaps_before),
            }
            (args.out / f"{args.mode}-soak.json").write_text(json.dumps(soak) + "\n")
            print(f"soak: {json.dumps(soak)}", flush=True)
        if shim is not None:
            link = body._link.stats()
            print(f"link: {json.dumps(link)}")
            print(f"shim: connections ended: {shim.ended}")
        table = summarise(args.mode, runs, cuts)
        (args.out / f"{args.mode}-table.md").write_text(table + "\n")
        print(table)
    finally:
        scorer.close()
        body.stop()
        if shim is not None:
            shim.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
