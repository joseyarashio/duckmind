"""Press play dead on a live check server and measure what the physics body did (2026-10-11).

    uv run python tools/measure_play_dead.py --port 8995 --body-port 53607 --mjcf DIR [--n 4] [--emotion play_dead] [--out FILE.jsonl]

Per press (``POST /api/neuro/motion {"id": "emo_play_dead"}``, the page's button): did it lie down (the trunk's tilt reached
about 70 degrees or more), were the legs up (the feet's height above the trunk's origin and above the floor, by the robot
model's forward kinematics on the body samples: joints, trunk position and orientation), did it get up by itself (the seconds
from the press until it stood again: robotd walk / stand, tilt under 25 degrees, held a second), was a righting needed or did
it stay down, and the impact numbers of the body host's ``x_contacts`` (the largest change of the trunk's velocity in one physics
step as an acceleration, its fastest turn, the largest contact force between the duck and anything else). Run the server with
``DUCKMIND_EMOTIONS=off`` so the brain's own emotions do not mix in. ``--emotion devastated`` presses the plain sit
for the comparison of the impact numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

JOINTS = (  # the body samples' ``j`` (FRAME_JOINTS): left leg, head, right leg (the mouth is not in it)
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)  # fmt: skip
ENDED = ("done", "fell", "interrupted")
LAY_TILT_DEG = 70.0
UP_TILT_DEG = 25.0


def get(port: int, path: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def post(port: int, path: str, body: dict[str, Any], timeout: float = 8.0) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )  # fmt: skip
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


class Host:
    """The body host's x_contacts (its own port, next to robotd's)."""

    def __init__(self, port: int) -> None:
        self.s = socket.create_connection(("127.0.0.1", port), timeout=3)
        self.f = self.s.makefile("rwb")

    def call(self, **params: Any) -> dict[str, Any]:
        self.f.write((json.dumps({"op": "x_contacts", **params}) + "\n").encode())
        self.f.flush()
        return json.loads(self.f.readline())


class Kinematics:
    """Forward kinematics of the robot model: where the feet are for one body sample."""

    def __init__(self, mjcf_dir: Path) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_dir / "robot_allcollisions.xml"))
        self.data = mujoco.MjData(self.model)
        self.qpos = {}
        for name in JOINTS:
            act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint = int(self.model.actuator_trnid[act, 0])
            self.qpos[name] = int(self.model.jnt_qposadr[joint])
        self.feet = [int(self.model.body(n).id) for n in ("ankle_left", "ankle_right")]
        self.trunk = int(self.model.body("trunk_base").id)

    def feet_z(self, frame: dict[str, Any]) -> tuple[float, float, float]:
        """(left foot z, right foot z, trunk origin z) in the world, metres."""
        m, d = self.model, self.data
        d.qpos[:] = 0.0
        d.qpos[0:3] = frame["p"]
        d.qpos[3:7] = frame["q"]
        for name, value in zip(JOINTS, frame["j"], strict=True):
            d.qpos[self.qpos[name]] = value
        self.mujoco.mj_kinematics(m, d)
        return (
            float(d.xpos[self.feet[0]][2]),
            float(d.xpos[self.feet[1]][2]),
            float(d.xpos[self.trunk][2]),
        )


def tilt_deg(q: list[float]) -> float:
    """The angle between the trunk's up axis and the world's up (degrees)."""
    w, x, y, z = q
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


class Capture:
    """Polls the body samples and the robotd label while a press runs."""

    def __init__(self, port: int) -> None:
        self.port, self.frames, self.labels = port, [], []
        self.stop = threading.Event()
        self.last_t: float | None = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                body = get(
                    self.port, f"/api/body?since_t={0 if self.last_t is None else self.last_t}"
                )
                new = [f for f in (body.get("frames") or []) if isinstance(f, dict)]
                for f in new:
                    if self.last_t is None or float(f["t"]) > self.last_t:
                        self.frames.append(f)
                        self.last_t = float(f["t"])
                if self.frames:
                    self.labels.append((self.last_t, body.get("policy"), bool(body.get("fallen"))))
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self) -> Capture:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(2)


def wait_running(port: int, limit_s: float = 240.0) -> None:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            if (
                get(port, "/api/neuro/telemetry?since=0").get("driver", {}).get("state")
                == "running"
            ):
                return
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("the driver did not come up")


def stand_still(port: int, limit_s: float = 60.0) -> bool:
    """A calm standing duck (the brain walks around between presses)."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        body = get(port, "/api/body")
        if (
            body.get("policy") in ("stand", "walk")
            and not body.get("fallen")
            and not body.get("held")
        ):
            tel = get(port, "/api/neuro/telemetry?since=0")
            if not (tel.get("motion", {}).get("emotion", {}).get("running")):
                time.sleep(1.0)
                return True
        time.sleep(0.5)
    return False


def analyse(kin: Kinematics | None, frames: list[dict[str, Any]], t_press: float) -> dict[str, Any]:
    """The lie-down in numbers, from the body samples after the press (sim seconds from the press)."""
    out: dict[str, Any] = {}
    run = [f for f in frames if float(f["t"]) >= t_press - 0.05]
    if not run:
        return out
    tilts = [(float(f["t"]) - t_press, tilt_deg(f["q"])) for f in run]
    out["tilt_max_deg"] = round(max(v for _, v in tilts), 1)
    out["lay_down"] = out["tilt_max_deg"] >= LAY_TILT_DEG
    down = [t for t, v in tilts if v >= LAY_TILT_DEG]
    out["lay_from_s"] = round(down[0], 2) if down else None
    if kin is not None and down:
        best_above, best_floor, fz = -9.0, -9.0, []
        for f in run:
            rel = float(f["t"]) - t_press
            if tilt_deg(f["q"]) < LAY_TILT_DEG:
                continue
            zl, zr, zt = kin.feet_z(f)
            best_above = max(best_above, min(zl, zr) - zt)
            best_floor = max(best_floor, min(zl, zr))
            fz.append((rel, min(zl, zr) - zt, min(zl, zr)))
        out["feet_above_trunk_m"] = round(best_above, 3)
        out["feet_above_floor_m"] = round(best_floor, 3)
        up = [t for t, above, _ in fz if above > 0.0]
        out["legs_up"] = bool(up)
        out["legs_up_from_s"] = round(up[0], 2) if up else None
        out["legs_up_s"] = round(up[-1] - up[0], 2) if len(up) > 1 else 0.0
    # up again: the first time after the lie it stands and the tilt is under 25 degrees for a second
    stood = None
    if down:
        t_lie = down[0]
        for f in run:
            rel = float(f["t"]) - t_press
            if rel < t_lie:
                continue
            if tilt_deg(f["q"]) < UP_TILT_DEG:
                if stood is None:
                    stood = rel
                elif rel - stood >= 1.0:
                    break
            else:
                stood = None
        else:
            stood = None if stood is None else stood
    out["up_s"] = None if stood is None else round(stood, 2)
    return out


def press_once(
    port: int, host: Host | None, kin: Kinematics | None, emotion: str, wait_s: float
) -> dict[str, Any]:
    eid = "emo_" + emotion
    stand_still(port)
    seq0 = (
        get(port, "/api/neuro/telemetry?since=0").get("motion", {}).get("emotion", {}).get("last")
        or {}
    ).get("seq", 0)
    if host is not None:
        host.call(do="start")
    cap = Capture(port)
    cap.__enter__()
    time.sleep(0.6)
    t_press = cap.last_t if cap.last_t is not None else 0.0
    wall0 = time.time()
    answer = post(port, "/api/neuro/motion", {"id": eid})
    row: dict[str, Any] = {
        "id": eid,
        "answer_ok": bool(answer.get("ok")),
        "reason": answer.get("reason"),
        "wall0": wall0,
    }
    if not answer.get("ok"):
        cap.__exit__()
        return row
    ended = None
    deadline = time.time() + wait_s
    down_read = False
    while time.time() < deadline:
        if host is not None and not down_read and time.time() - wall0 >= 7.0:
            row["impact_down"] = host.call(do="read")  # the lie-down alone (the wake is after 10 s)
            down_read = True
        tel = get(port, "/api/neuro/telemetry?since=0")
        last = (tel.get("motion", {}).get("emotion") or {}).get("last") or {}
        if last.get("seq", 0) > seq0 and last.get("state") in ENDED:
            ended = last
            break
        time.sleep(0.25)
    row["wall_s"] = round(time.time() - wall0, 2)
    row["ended"] = ended
    t0 = time.time()
    final: dict[str, Any] | None = None
    while (
        time.time() - t0 < 25.0
    ):  # after the play: standing within 25 s, upright (a righting may be under way)
        body = get(port, "/api/body")
        final = {"policy": body.get("policy"), "fallen": body.get("fallen")}
        if body.get("policy") in ("stand", "walk") and not body.get("fallen"):
            break
        time.sleep(0.5)
    row["final"] = final
    row["stood_again"] = bool(
        final and final["policy"] in ("stand", "walk") and not final["fallen"]
    )
    if host is not None:
        row["impact"] = host.call(do="read")
        host.call(do="stop")
    cap.__exit__()
    row["frames"] = [f for f in cap.frames if isinstance(f, dict)]
    row["t_press"] = t_press
    row["lay"] = analyse(kin, row["frames"], t_press)
    row["policies"] = [p for _, p, _ in cap.labels][::5][:200]
    tel = get(port, "/api/neuro/telemetry?since=0")
    row["righting_events"] = [
        e.get("kind") for e in (tel.get("events") or [])[-60:] if "righting" in str(e.get("kind"))
    ]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8995)
    ap.add_argument("--body-port", type=int, default=0)
    ap.add_argument("--mjcf", default="")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--emotion", default="play_dead")
    ap.add_argument("--wait", type=float, default=60.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--dump", default="", help="write each press's body samples to FILE.<i>.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wait_running(args.port)
    host = Host(args.body_port) if args.body_port else None
    kin = Kinematics(Path(args.mjcf)) if args.mjcf else None
    out = open(args.out, "a") if args.out else sys.stdout
    for i in range(args.n):
        row = press_once(args.port, host, kin, args.emotion, args.wait)
        row["i"], row["label"] = i, args.label
        frames = row.pop("frames", None)
        if args.dump and frames:
            Path(f"{args.dump}.{i}.json").write_text(
                json.dumps({"t_press": row.get("t_press"), "frames": frames})
            )
        print(json.dumps(row, ensure_ascii=False), file=out, flush=True)


if __name__ == "__main__":
    main()
