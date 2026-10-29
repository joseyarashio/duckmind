"""hop_over against a real sill on a live check server (bug 7, 2026-10-09).

    python tools/make_sill_home.py /tmp/sill_home.json
    DUCKMIND_LIVE_HOME=/tmp/sill_home.json DUCKMIND_EMOTIONS=off  <the usual launch>  (e.g. --port 8989)
    python tools/measure_hop_sill.py --port 8989 --body-port <robotd --sim port> --n 5 --out rows.jsonl

Per trial: the body host puts the duck (STAND) --near-mm in front of a sill's near face (default 120 mm) facing it (``x_place``), the page's button
``hop_over_<cm>`` is pressed once it stands calm (POST /api/neuro/motion, the table's guard decides), the body samples
are captured for 7 s. Row: did it start (and why not), the furthest the trunk got past the sill's far edge, whether it
ended on the far side, whether it fell (trunk tilt over 60 degrees or the page's ``fallen``), the trunk's closest
approach and peak tilt. A press that is refused is a row too (``started: false``).
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
import urllib.request

SILLS = {
    2: (-1500.0, 1000.0, +1.0),
    4: (-1500.0, -1000.0, -1.0),
}  # cm -> (x mm, y mm, direction to walk: +1 is +y)
DEPTH_MM = 40.0
START_BEFORE_MM = (
    120.0  # the trunk centre's start, this far before the sill's near face (training: 60-180 mm)
)
CAPTURE_S = 7.0


def get(port: int, path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def post(port: int, path: str, body: dict, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Host:
    def __init__(self, port: int) -> None:
        self.f = socket.create_connection(("127.0.0.1", port), timeout=5).makefile("rwb")

    def call(self, op: str, **params) -> dict:
        self.f.write((json.dumps({"op": op, **params}) + "\n").encode())
        self.f.flush()
        return json.loads(self.f.readline())


def stand_calm(port: int, limit_s: float = 90.0, hold_s: float = 2.0) -> bool:
    """Wait until the body has been standing or walking, upright and not held, for ``hold_s`` in a row (after a fall the
    righting takes its time and robotd's fallen flag outlasts the place)."""
    t0 = time.time()
    since = None
    while time.time() - t0 < limit_s:
        b = get(port, "/api/body")
        calm = b.get("policy") in ("stand", "walk") and not b.get("fallen") and not b.get("held")
        if calm:
            since = since or time.time()
            if time.time() - since >= hold_s:
                return True
        else:
            since = None
        time.sleep(0.3)
    return False


def trial(port: int, host: Host, cm: int, i: int, near_mm: float = START_BEFORE_MM) -> dict:
    sx, sy, d = SILLS[cm]
    start_y = sy - d * (DEPTH_MM / 2 + near_mm)
    yaw = math.pi / 2 if d > 0 else -math.pi / 2
    placed = host.call(
        "x_place", x_mm=round(sx), y_mm=round(start_y), yaw_mrad=round(yaw * 1000), keyframe="STAND"
    )
    row: dict = {
        "cm": cm,
        "i": i,
        "near_mm": near_mm,
        "placed": placed.get("ok"),
        "wall0": time.time(),
    }
    row["calm"] = stand_calm(port)
    # (the brain walks on while the body stood calm: put it back, then press at once)
    host.call(
        "x_place", x_mm=round(sx), y_mm=round(start_y), yaw_mrad=round(yaw * 1000), keyframe="STAND"
    )
    time.sleep(0.15)
    pre = (host.call("x_world").get("duck") or {}).get("pos_mm")
    row["pre_press_mm"] = [round(float(pre[0])), round(float(pre[1]))] if pre else None
    samples: list[
        tuple[float, float, float, float]
    ] = []  # (wall, x mm, y mm, tilt mrad): the world's truth
    t_press = time.time()
    ans = post(port, "/api/neuro/motion", {"id": f"hop_over_{cm}"})
    row.update(
        started=bool(ans.get("ok")), reason=ans.get("reason"), reason_zh=ans.get("reason_zh")
    )
    while time.time() - t_press < CAPTURE_S:
        duck = host.call("x_world").get("duck") or {}
        pos = duck.get("pos_mm")
        if pos:
            samples.append(
                (time.time(), float(pos[0]), float(pos[1]), float(duck.get("tilt_mrad", 0.0)))
            )
        time.sleep(0.05)
    if samples:
        # progress along the walking direction, from the sill's centre line (mm): the far edge is +DEPTH/2
        prog = [d * (y - sy) for _, _, y, _ in samples]
        row["start_prog_mm"] = round(prog[0], 1)
        row["max_prog_mm"] = round(max(prog), 1)
        row["end_prog_mm"] = round(prog[-1], 1)
        row["sideways_mm"] = round(max(abs(x - sx) for _, x, _, _ in samples), 1)
        row["crossed"] = bool(
            max(prog) >= DEPTH_MM / 2 + 30.0
        )  # the trunk centre 3 cm past the far edge
        row["ended_far_side"] = bool(prog[-1] >= DEPTH_MM / 2)
        tilts = [t / 1000.0 * 180.0 / math.pi for _, _, _, t in samples]
        row["peak_tilt_deg"] = round(max(tilts), 1)
        row["fell"] = bool(max(tilts) > 60.0)
    pts = samples
    row["frames"] = len(pts)
    body = get(port, "/api/body")
    row["final"] = {"policy": body.get("policy"), "fallen": body.get("fallen")}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--body-port", type=int, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--cms", default="2,4")
    ap.add_argument(
        "--near-mm",
        default="120",
        help="trunk centre to the near face, comma list (training: 60-180)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    host = Host(args.body_port)
    out = open(args.out, "a") if args.out else sys.stdout
    for near in (float(v) for v in args.near_mm.split(",")):
        for cm in (int(c) for c in args.cms.split(",")):
            for i in range(args.n):
                row = trial(args.port, host, cm, i, near)
                print(json.dumps(row, ensure_ascii=False), file=out, flush=True)


if __name__ == "__main__":
    main()
