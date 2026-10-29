"""Press the page's 動作試玩 buttons on a live check server and read what the physics body did (2026-10-09).

    uv run python tools/measure_motions.py --port 8986 --ids quick_turn_left,quick_turn_right [--n 5] [--out FILE.jsonl]

Per press (``POST /api/neuro/motion {"id": ...}``, the same as the page's button): did it start, what the server said it did
(``motion.last`` of ``/api/neuro/telemetry``: seconds, degrees turned, travel, the verdict), whether the duck fell, and
whether it stood again afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def get(port: int, path: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def post(port: int, path: str, body: dict, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def motion(port: int) -> dict:
    return get(port, "/api/neuro/telemetry").get("motion") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else None
    rows = []
    for mid in a.ids.split(","):
        for k in range(a.n):
            # a calm moment: nothing running, the last result older than a few seconds
            for _ in range(60):
                m = motion(a.port)
                if m.get("running") is None:
                    break
                time.sleep(1)
            time.sleep(2.0)
            before = (motion(a.port).get("last") or {}).get("seq", 0)
            t0 = time.time()
            tries = 0
            while True:  # the page's button is pressed again by a person when the duck was walking too fast: up to 8 tries
                tries += 1
                try:
                    ans = post(a.port, "/api/neuro/motion", {"id": mid})
                except Exception as exc:  # noqa: BLE001
                    ans = {"ok": False, "error": str(exc)}
                if (
                    ans.get("started")
                    or tries >= 8
                    or ans.get("reason") not in ("speed", "settle", "gap", "busy")
                ):
                    break
                time.sleep(3)
            row = {"id": mid, "press": k, "answer": ans, "tries": tries}
            if ans.get("ok") and ans.get("started"):
                for _ in range(40):
                    last = motion(a.port).get("last") or {}
                    if last.get("seq", 0) > before:
                        row["result"] = {
                            key: last.get(key) for key in ("verdict", "numbers", "zh", "tail_zh")
                        }
                        break
                    time.sleep(0.5)
                row["wall_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False)[:400], flush=True)
            if out:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    started = sum(1 for r in rows if r["answer"].get("started"))
    print(f"{started}/{len(rows)} started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
