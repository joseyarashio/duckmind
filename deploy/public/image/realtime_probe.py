#!/usr/bin/env python3
"""How fast a public duck's simulation runs against the wall clock (the page's own number).

    realtime_probe.py URL [--minutes 10] [--interval 5] [--cookie 'dm_session=…'] [--out FILE.jsonl]
                      [--ready-timeout 300] [--page-load]

URL is a duck server (``http://127.0.0.1:8080`` for a local container, or the public site for a
live session: pass that session's ``dm_session`` cookie). The probe

1. waits for ``GET /api/ping`` to answer 200 and for ``/api/neuro``'s ``monitor.realtime`` to be a
   number (start-up: seconds from the probe's start to each),
2. then reads ``/api/neuro`` every INTERVAL seconds for MINUTES minutes and keeps its ``monitor``
   (``realtime`` = simulated seconds per wall second over the last 10 s; ``robotd_hz``; stalls),
3. prints one JSON summary: n, median / p5 / min realtime, the share of samples at or above 0.95
   (duckmind's own bar: a run under it is not a fair run), robotd's slowest second, and the longest
   physics stall. Exit status 0 when the median is at least 0.95 and at most 5 % of samples are under
   it, else 1.

``--page-load`` also polls what an open page polls (measured on the page, 2026-11-10: ``/api/body``
4 a second, ``/api/neuro/sound/status`` and ``/api/eye.jpg`` every 2 s, ``/api/neuro/knows`` every
10 s), so a duck nobody watches costs what a watched one does; leave it off when a real browser has
the page open.

Start it on the first live start of a new machine: amd64 and arm64 cores differ, so the numbers
measured on an arm64 Mac only say what to try first. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

BAR = 0.95


def fetch(url: str, cookie: str | None, timeout: float = 5.0) -> tuple[int, Any]:
    headers = {"User-Agent": "duckmind-realtime-probe"}
    if cookie:
        headers["Cookie"] = cookie
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, None
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, None
    try:
        return status, json.loads(body)
    except ValueError:
        return status, None


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


# what an open page polls (path, period in seconds)
PAGE_POLLS = (
    ("/api/body", 0.25),
    ("/api/neuro/sound/status", 2.0),
    ("/api/eye.jpg", 2.0),
    ("/api/neuro/knows", 10.0),
)


def page_load(base: str, cookie: str | None, stop: threading.Event) -> None:
    """Poll :data:`PAGE_POLLS` until ``stop`` is set."""
    due = {path: 0.0 for path, _ in PAGE_POLLS}
    while not stop.is_set():
        now = time.monotonic()
        for path, period in PAGE_POLLS:
            if now >= due[path]:
                due[path] = now + period
                fetch(base + path, cookie)
        stop.wait(0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--cookie", default=None)
    parser.add_argument("--out", default=None, help="append every sample here (JSON lines)")
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--page-load", action="store_true", help="poll like an open page does")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")
    out = open(args.out, "a", encoding="utf-8") if args.out else None  # noqa: SIM115

    t0 = time.monotonic()
    ping_s = running_s = None
    while time.monotonic() - t0 < args.ready_timeout:
        if ping_s is None and fetch(base + "/api/ping", args.cookie)[0] == 200:
            ping_s = time.monotonic() - t0
        if ping_s is not None:
            status, body = fetch(base + "/api/neuro", args.cookie)
            monitor = (body or {}).get("monitor") or {}
            if status == 200 and isinstance(monitor.get("realtime"), (int, float)):
                running_s = time.monotonic() - t0
                break
        time.sleep(1.0)
    if running_s is None:
        print(json.dumps({"ready": False, "ping_s": ping_s, "after_s": args.ready_timeout}))
        return 1

    stop = threading.Event()
    if args.page_load:
        threading.Thread(target=page_load, args=(base, args.cookie, stop), daemon=True).start()
    samples: list[dict[str, Any]] = []
    end = time.monotonic() + args.minutes * 60.0
    while time.monotonic() < end:
        status, body = fetch(base + "/api/neuro", args.cookie)
        monitor = (body or {}).get("monitor") or {}
        row = {"t": round(time.time(), 1), "status": status, "state": (body or {}).get("state")}
        row.update({k: monitor.get(k) for k in ("realtime", "robotd_hz", "robotd_hz_min_10s",
                                                "max_stall_ms_10s", "trust")})  # fmt: skip
        samples.append(row)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
        time.sleep(args.interval)
    stop.set()

    rts = [s["realtime"] for s in samples if isinstance(s.get("realtime"), (int, float))]
    hz = [
        s["robotd_hz_min_10s"]
        for s in samples
        if isinstance(s.get("robotd_hz_min_10s"), (int, float))
    ]
    stalls = [
        s["max_stall_ms_10s"]
        for s in samples
        if isinstance(s.get("max_stall_ms_10s"), (int, float))
    ]
    summary = {
        "ready": True,
        "ping_s": None if ping_s is None else round(ping_s, 1),
        "realtime_ready_s": round(running_s, 1),
        "minutes": args.minutes,
        "n": len(rts),
        "missing": len(samples) - len(rts),
        "realtime_median": round(statistics.median(rts), 3) if rts else None,
        "realtime_p5": round(quantile(rts, 0.05), 3) if rts else None,
        "realtime_min": round(min(rts), 3) if rts else None,
        "share_at_bar": round(sum(r >= BAR for r in rts) / len(rts), 3) if rts else None,
        "robotd_hz_min": min(hz) if hz else None,
        "stall_ms_max": max(stalls) if stalls else None,
        "suspect_share": round(sum(s.get("trust") == "suspect" for s in samples) / len(samples), 3)
        if samples else None,
    }  # fmt: skip
    print(json.dumps(summary))
    good = bool(rts) and summary["realtime_median"] >= BAR and summary["share_at_bar"] >= 0.95
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
