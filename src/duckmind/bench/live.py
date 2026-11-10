"""The live mode: the same click set on a running test server, in real time — for the last confirmation of a decision the
headless runner has already made, never for the search.

A live duck cannot be put at a click's start pose (its lasting map would no longer know where it is), so the set is used
the other way round: before each click the duck's true pose is read, and the unused click whose start is nearest is
asked (within :data:`START_TOL_M`); when the duck stands in clutter or no start is near, it is first sent to the nearest
unused start by an errand that is not scored. The row carries the click's ``k`` and is labelled again from the pose the
duck really started from (class, what is on the line, reachability), so a click that stopped being a fair question from
there is not asked.

The scene is checked before every click as every live measurement did (daytime, battery, not afraid, not fallen, nobody
holding it, no visit), and the body's realtime factor is sampled through the click: a click with a sample under
:data:`REALTIME_MIN` is written to ``<out>.discarded.jsonl`` and asked again later (the machine was too busy to trust it).

Rows have the headless runner's schema (``mode`` "live"). Without the server's 50 Hz record (``--rec``) brushes and the
body link's own ``blocked`` stops are not visible and stay None.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from duckmind.bench import clicks as C
from duckmind.bench import runner as RN

START_TOL_M = 0.35  # a click is asked from a pose this near its start
START_CLEAR_M = 0.20  # … with at least this much room there
REALTIME_MIN = 0.95
FERRIES_MAX = 4
TRIES_MAX = 2
NOT_AFRAID = ("hide", "flee")
END = ("arrived", "gave_up")


class Server:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.base + path, timeout=20) as r:
            return json.load(r)  # type: ignore[no-any-return]

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)  # type: ignore[no-any-return]
        except urllib.error.HTTPError as e:
            return json.load(e)  # type: ignore[no-any-return]

    def scene(self) -> dict[str, Any]:
        """The fair-scene facts from the page's own answers (truth only to set and report the scene)."""
        neuro = self.get("/api/neuro?since=999999999")
        body = self.get("/api/body")
        brain, frame = neuro.get("brain") or {}, neuro.get("frame") or {}
        world = body.get("world") or {}
        truth = (body.get("frame") or {}).get("truth") or {}
        return {
            "battery": float((frame.get("power") or {}).get("battery") or 0.0),
            "fear": float((brain.get("drives") or {}).get("fear") or 0.0),
            "behaviour": str(brain.get("behaviour") or ""),
            "posture": str((frame.get("body") or {}).get("posture") or ""),
            "fallen": bool((frame.get("body") or {}).get("fallen")),
            "asleep": bool(neuro.get("asleep")),
            "holding": bool((world.get("hand") or {}).get("holding")),
            "visits": list((world.get("family") or {}).get("visits") or ()),
            "day": float((neuro.get("day") or {}).get("minute") or 0.0),
            "realtime": body.get("realtime"),
            "truth": (
                truth.get("x_mm", 0) / 1000.0,
                truth.get("y_mm", 0) / 1000.0,
                truth.get("yaw_mrad", 0) / 1000.0,
            ),
        }


def fair(s: dict[str, Any]) -> bool:
    return (
        s["battery"] >= 0.5
        and s["fear"] < 0.3
        and s["behaviour"] not in NOT_AFRAID
        and not s["fallen"]
        and not s["asleep"]
        and not s["holding"]
        and not s["visits"]
        and 480.0 <= s["day"] <= 1080.0
        and s["posture"] in ("stand", "sit")
    )


def wait_fair(
    server: Server, limit_s: float = 120.0, sleep: Any = time.sleep
) -> dict[str, Any] | None:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        s = server.scene()
        if not (480.0 <= s["day"] <= 1080.0):
            server.post("/api/neuro/clock", {"minute": 600})
        elif fair(s) and isinstance(s["realtime"], (int, float)) and s["realtime"] >= REALTIME_MIN:
            return s
        sleep(1.0)
    return None


def choose(
    lab: C.Labeller, pose: Sequence[float], pool: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """``(click, its labels from this pose, the nearest start to go to instead)``: the unused click nearest ``pose`` that is
    still the same fair question from here, else where to send the duck first."""
    x, y = float(pose[0]), float(pose[1])
    near = sorted(pool, key=lambda c: math.hypot(c["pose"][0] - x, c["pose"][1] - y))
    if not near:
        return None, None, None
    if lab.fm.clear_at(x, y) >= START_CLEAR_M:
        for c in near:
            if math.hypot(c["pose"][0] - x, c["pose"][1] - y) > START_TOL_M:
                break
            info = lab.label(pose, c["point"])
            if (
                info["reach"]
                and info["cls"] == c["cls"]
                and not (set(info["flags"]) - {C.FLAG_CLUTTER})
            ):
                return c, info, None
    return None, None, near[0]


def run_click(
    server: Server, x: float, y: float, secs: float, rec: str | None, sleep: Any = time.sleep
) -> dict[str, Any]:
    """Send the duck to (x, y) and watch it: the errand's end, falls, bumps, the realtime factor."""
    ans = server.post("/api/neuro/gothere", {"point": {"x_m": x, "y_m": y}})
    out: dict[str, Any] = {
        "ok": bool(ans.get("ok")),
        "refused": ans.get("reason"),
        "range_m": ans.get("range_m"),
        "bearing": ans.get("bearing_rad"),
    }
    if not out["ok"]:
        return out
    t0 = time.time()
    seq = int(server.get("/api/neuro/telemetry?since=0").get("seq") or 0)
    state, arrived_s, last = "waiting", None, {}
    falls = bumps = startles = takeovers = 0
    blocked = brushes = 0
    was_fallen = False
    was_walk = False
    rts: list[float] = []
    truth = None
    f = open(rec, "rb") if rec else None  # noqa: SIM115
    carry = b""
    if f is not None:
        f.seek(0, 2)
    try:
        while time.time() - t0 < secs:
            d = server.get(f"/api/neuro/telemetry?since={seq}")
            seq = int(d.get("seq") or seq)
            last = d.get("errand") or {}
            state = str(last.get("state") or state)
            beh = d.get("behaviour")
            beh = beh.get("key") if isinstance(beh, dict) else beh
            if state == "walking":
                takeovers += int(was_walk and beh != "wander")
                was_walk = beh == "wander"
            for e in d.get("events") or ():
                ref = str(e.get("reflex") or "")
                bumps += ref == "bump_stop"
                startles += ref == "startle"
            body = server.get("/api/body")
            if isinstance(body.get("realtime"), (int, float)):
                rts.append(float(body["realtime"]))
            tr = (body.get("frame") or {}).get("truth") or {}
            if tr:
                truth = (tr.get("x_mm", 0) / 1000.0, tr.get("y_mm", 0) / 1000.0)
            if f is None:
                fallen = bool(
                    (
                        (server.get("/api/neuro?since=999999999").get("frame") or {}).get("body")
                        or {}
                    ).get("fallen")
                )
                falls += int(fallen and not was_fallen)
                was_fallen = fallen
            else:
                lines = (carry + f.read()).split(b"\n")
                carry = lines.pop()
                for line in lines:
                    try:
                        fr = (json.loads(line).get("frame")) or {}
                    except ValueError:
                        continue
                    fallen = bool((fr.get("body") or {}).get("fallen"))
                    falls += int(fallen and not was_fallen)
                    was_fallen = fallen
                    for ev in fr.get("events") or ():
                        blocked += ev == "blocked"
                        brushes += ev == "brushed"
            if state == "arrived" and arrived_s is None:
                arrived_s = round(time.time() - t0, 1)
            if state in END:
                break
            sleep(0.5)
    finally:
        if f is not None:
            f.close()
    reason = (last.get("last") or {}).get("reason") if state == "gave_up" else None
    out.update(
        state=state,
        reason=reason or last.get("reason") if state == "gave_up" else None,
        arrived=state == "arrived" and arrived_s is not None and arrived_s <= secs,
        arrived_s=arrived_s,
        secs=round(time.time() - t0, 1),
        truth_end=None if truth is None else round(math.hypot(x - truth[0], y - truth[1]), 2),
        falls=falls,
        bumps=bumps + blocked,
        brushes=brushes if rec else None,
        startles=startles,
        gave_up=state == "gave_up",
        takeovers=takeovers,
        rt_min=round(min(rts), 3) if rts else None,
        load=round(os.getloadavg()[0], 2),
    )
    return out


def run_live(
    base: str,
    clicks: Sequence[dict[str, Any]],
    out: Path,
    *,
    home: str = "real_home",
    rec: str | None = None,
    tree: str | None = None,
    secs: float = RN.CLICK_S,
    max_clicks: int = 0,
    log: Any = print,
    sleep: Any = time.sleep,
) -> list[dict[str, Any]]:
    """Ask ``clicks`` of the server at ``base`` (each once, nearest start first) and append their rows to ``out``."""
    from duckmind.world import homelayout

    server = Server(base)
    lab = C.Labeller(homelayout.load(home))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in RN.load_rows(out) if r.get("mode") == "live"] if out.is_file() else []
    done = {int(r["k"]) for r in rows}
    tries: dict[int, int] = {}
    pool = [c for c in clicks if int(c["k"]) not in done]
    ferries = 0
    with (
        out.open("a", encoding="utf-8") as f,
        out.with_suffix(".discarded.jsonl").open("a", encoding="utf-8") as bad,
    ):
        asked = 0
        while pool and not (max_clicks and asked >= max_clicks):
            s = wait_fair(server, sleep=sleep)
            if s is None:
                log("the scene is not fair or the server is slow: stopped")
                break
            click, info, go = choose(lab, s["truth"], pool)
            if click is None or info is None:
                if go is None or ferries >= FERRIES_MAX:
                    log(
                        f"no click can be asked from {tuple(round(v, 2) for v in s['truth'])}: stopped"
                    )
                    break
                ferries += 1
                run_click(
                    server, go["pose"][0], go["pose"][1], secs, None, sleep
                )  # (not scored: getting to a start)
                sleep(1.5)
                continue
            ferries = 0
            got = run_click(server, click["point"][0], click["point"][1], secs, rec, sleep)
            k = int(click["k"])
            row: dict[str, Any] = dict.fromkeys(RN.ROW_FIELDS)
            row.update(schema=RN.SCHEMA, set=click.get("set"), k=k, arm="main", mode="live", tree=tree, seed=None,
                       pose=[round(v, 3) for v in s["truth"]], point=click["point"], cls=info["cls"], line=info["line"],
                       flagged=bool(info["flags"]), detour=info["detour"],
                       start_off_m=round(math.hypot(click["pose"][0] - s["truth"][0], click["pose"][1] - s["truth"][1]), 2))  # fmt: skip
            row.update({key: v for key, v in got.items() if key not in ("ok", "refused")})
            row["truth_ok"] = (
                None
                if got.get("truth_end") is None
                else bool(got.get("arrived") and got["truth_end"] <= RN.TRUTH_OK_M)
            )
            slow = got.get("ok") and (got.get("rt_min") is None or got["rt_min"] < REALTIME_MIN)
            if not got.get("ok") or slow:
                tries[k] = tries.get(k, 0) + 1
                bad.write(
                    json.dumps({**row, "refused": got.get("refused")}, ensure_ascii=False) + "\n"
                )
                bad.flush()
                if tries[k] >= TRIES_MAX:
                    pool = [c for c in pool if int(c["k"]) != k]
                log(
                    f"click {k} discarded ({'refused: ' + str(got.get('refused')) if not got.get('ok') else 'realtime ' + str(got.get('rt_min'))})"
                )
                sleep(5.0)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            asked += 1
            pool = [c for c in pool if int(c["k"]) != k]
            if log is not None and len(rows) % 10 == 0:
                log(f"[live] {len(rows)} clicks, arrived {sum(1 for r in rows if r['arrived'])}")
            sleep(1.5)
    return rows


__all__ = [
    "REALTIME_MIN",
    "START_TOL_M",
    "Server",
    "choose",
    "fair",
    "run_click",
    "run_live",
    "wait_fair",
]
