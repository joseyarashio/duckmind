"""Press the page's emotions on a live check server and measure what the physics body did (2026-10-08).

    uv run python tools/measure_emotions.py --port 8983 --body-port 56671 [--n 5] [--only sad,yes] [--out FILE.jsonl]

Per press (``POST /api/neuro/motion {"id": "emo_<name>"}``, the same as the page's button): did it start, run to the end,
fall, how long (played / wall), the head's and the beak's contacts with the floor or the duck (the body host's
``x_contacts``), how far the trunk moved and tilted, whether the duck stood (or, for a sit emotion, stood up again) in
the end, and the sound against the beak: the sim time of the body sample that carried the sound's call (``v``) and of the
beak's first opening (``m`` > 0.15) — the page plays the sound as that sample is drawn. Run the server with
``DUCKMIND_EMOTIONS=off`` so the brain's own emotions do not mix in.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
import urllib.request
import wave
from pathlib import Path

from duckmind.neuro import emotions as EO

ENDED = ("done", "fell", "interrupted")


def get(port: int, path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def post(port: int, path: str, body: dict, timeout: float = 8.0) -> dict:
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

    def call(self, **params) -> dict:
        self.f.write((json.dumps({"op": "x_contacts", **params}) + "\n").encode())
        self.f.flush()
        return json.loads(self.f.readline())


def wav_onset_s(path: Path, thr_db: float = -40.0) -> float | None:
    if not path.exists():
        return None
    with wave.open(str(path)) as w:
        import array

        raw = array.array("h")
        raw.frombytes(w.readframes(w.getnframes()))
        sr = w.getframerate()
    peak = max(abs(v) for v in raw) or 1
    floor = peak * 10 ** (thr_db / 20)
    for i, v in enumerate(raw):
        if abs(v) >= floor:
            return i / sr
    return None


def wav_envelope(path: Path, hop_s: float = 0.01):
    import array

    import numpy as np

    with wave.open(str(path)) as w:
        raw = array.array("h")
        raw.frombytes(w.readframes(w.getnframes()))
        sr = w.getframerate()
    x = np.array(raw, dtype=float)
    h = int(sr * hop_s)
    rms = np.array([np.sqrt(np.mean(x[i * h : (i + 1) * h] ** 2)) for i in range(len(x) // h)])
    return rms / (rms.max() + 1e-9)


def beak_lag_ms(beak: list[tuple[float, float]], audio, hop_s: float = 0.01) -> int | None:
    """How late the measured beak follows the sound, by cross-correlation of the beak's opening with the sound's
    loudness (both at 10 ms; the sound starts at the voice sample, t = 0)."""
    import numpy as np

    if len(beak) < 10:
        return None
    t = np.arange(0, 12.0, hop_s)
    m = np.interp(t, [b[0] for b in beak], [b[1] for b in beak], left=0.0, right=0.0)
    a = np.zeros(len(t))
    a[: min(len(audio), len(t))] = audio[: len(t)]
    best = (-2.0, 0)
    for lag in range(-30, 46):
        shifted = np.roll(a, lag) if lag >= 0 else np.roll(a, lag)
        if lag > 0:
            shifted[:lag] = 0
        c = float(np.corrcoef(shifted, m)[0, 1]) if shifted.std() > 0 and m.std() > 0 else -2.0
        if c > best[0]:
            best = (c, lag)
    return best[1] * 10


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


def frames_since(port: int, since_t: float | None) -> list[dict]:
    q = "" if since_t is None else f"?since_t={since_t}"
    body = get(port, f"/api/body{q}")
    return list((body.get("frame") or {}).get("frames") or body.get("frames") or [])


class Capture:
    """Polls the body samples while a press runs (the page's ring keeps only the last second or so)."""

    def __init__(self, port: int) -> None:
        import threading

        self.port, self.frames, self.stop = port, [], threading.Event()
        self.last_t: float | None = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                q = f"?since_t={0 if self.last_t is None else self.last_t}"
                new = list(get(self.port, f"/api/body{q}").get("frames") or [])
                for f in new:
                    if self.last_t is None or float(f["t"]) > self.last_t:
                        self.frames.append(f)
                        self.last_t = float(f["t"])
            except Exception:
                pass
            time.sleep(0.25)

    def __enter__(self) -> Capture:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(2)


def stand_still(port: int, limit_s: float = 40.0) -> bool:
    """Wait for a calm standing duck (the brain walks around between presses)."""
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
                return True
        time.sleep(0.5)
    return False


def press_once(port: int, host: Host | None, name: str, voice_dir: Path | None) -> dict:
    eid = EO.ID_PREFIX + name
    e = EO.catalog()[name]
    stand_still(port)
    last_seq = (
        get(port, "/api/neuro/telemetry?since=0").get("motion", {}).get("emotion", {}).get("last")
        or {}
    ).get("seq", 0)
    if host is not None:
        host.call(do="start")
    cap = Capture(port)
    cap.__enter__()
    wall0 = time.time()
    answer = post(port, "/api/neuro/motion", {"id": eid})
    row: dict = {"id": eid, "name": name, "answer_ok": bool(answer.get("ok")), "reason": answer.get("reason"),
                 "duration_s": e.duration_s, "wall0": wall0}  # fmt: skip
    if not answer.get("ok"):
        cap.__exit__()
        return row
    deadline = time.time() + e.duration_s + (EO.SIT_HOLD_S if e.sits else 0) + 6.0
    ended = None
    while time.time() < deadline:
        tel = get(port, "/api/neuro/telemetry?since=0")
        last = (tel.get("motion", {}).get("emotion") or {}).get("last") or {}
        if last.get("seq", 0) > last_seq and last.get("state") in ENDED:
            ended = last
            break
        time.sleep(0.25)
    row["wall_s"] = round(time.time() - wall0, 2)
    row["ended"] = ended
    if host is not None:
        row["contacts"] = host.call(do="read")
        host.call(do="stop")
    # after: stands (or stands up again) within 14 s, upright
    t0 = time.time()
    final = None
    while time.time() - t0 < 14.0:
        body = get(port, "/api/body")
        final = {"policy": body.get("policy"), "fallen": body.get("fallen")}
        if body.get("policy") in ("stand", "walk") and not body.get("fallen"):
            break
        time.sleep(0.5)
    row["final"] = final
    row["stood_again"] = bool(
        final and final["policy"] in ("stand", "walk") and not final["fallen"]
    )
    cap.__exit__()
    frames = [f for f in cap.frames if isinstance(f, dict)]
    voiced = [
        f
        for f in frames
        if f.get("v") and any(str(c.get("tag", "")).startswith("emo_") for c in f["v"])
    ]
    row["frames"] = len(frames)
    if voiced:
        t_v = float(voiced[0]["t"])
        row["voice_t"] = 0.0
        after = [f for f in frames if float(f["t"]) >= t_v]
        end_t = t_v + e.duration_s
        run = [f for f in after if float(f["t"]) <= end_t + 0.5]
        opened = [f for f in run if f.get("m") is not None and float(f["m"]) > 0.15]
        if opened:
            row["beak_t"] = round(float(opened[0]["t"]) - t_v, 3)
            row["beak_peak"] = round(max(float(f["m"]) for f in run if f.get("m") is not None), 3)
        key_open = next((r[0] for r in e.rows if r[6] > 0.15), None)
        if voice_dir is not None:
            beak = [(float(f["t"]) - t_v, float(f["m"])) for f in run if f.get("m") is not None]
            row["beak_lag_ms"] = beak_lag_ms(
                beak,
                wav_envelope(
                    voice_dir / f"{EO.SOUND_PREFIX}{name}" / f"{EO.SOUND_PREFIX}{name}_a.wav"
                ),
            )
        row["key_beak_t"] = key_open
        pitches = []
        for f in run:
            q = f.get("q")
            if q:
                w, x, y, z = (float(v) for v in q)
                pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))))
        if pitches:
            base = pitches[0]
            row["trunk_pitch_range_deg"] = round(max(pitches) - min(pitches), 2)
            row["trunk_pitch_peak_deg"] = round(max(abs(v - base) for v in pitches), 2)
        row["key_bow_deg"] = round(math.degrees(max(r[5] for r in e.rows)), 1)
        tr = [f["p"] for f in run if f.get("p")]
        if tr:
            row["drift_m"] = round(max(math.hypot(p[0] - tr[0][0], p[1] - tr[0][1]) for p in tr), 3)
        # the head: the measured joints against the keyframes (neck, head_pitch, yaw, roll are joints 5..8 of the 14)
        errs = []
        home = (0.3491, 0.3491, 0.0, 0.0)
        for f in run:
            j = f.get("j")
            if not j:
                continue
            rel = float(f["t"]) - t_v
            if rel < 0 or rel > e.duration_s:
                continue
            row_k = EO._lerp_rows(e.rows, rel)
            for idx, hm in zip((5, 6, 7, 8), home, strict=True):
                errs.append(abs((float(j[idx]) - hm) - float(row_k[idx - 4])))
        if errs:
            row["head_err_mean_rad"] = round(sum(errs) / len(errs), 3)
    if voice_dir is not None:
        onset = wav_onset_s(
            voice_dir / f"{EO.SOUND_PREFIX}{name}" / f"{EO.SOUND_PREFIX}{name}_a.wav"
        )
        row["wav_onset_s"] = None if onset is None else round(onset, 3)
        if onset is not None and row.get("beak_t") is not None:
            row["beak_minus_audio_ms"] = round((row["beak_t"] - onset) * 1000)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8983)
    ap.add_argument("--body-port", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--voice-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    wait_running(args.port)
    host = Host(args.body_port) if args.body_port else None
    names = [n for n in EO.ORDER if not args.only or n in args.only.split(",")]
    out = open(args.out, "a") if args.out else sys.stdout
    voice_dir = Path(args.voice_dir) if args.voice_dir else None
    for name in names:
        for i in range(args.n):
            row = press_once(args.port, host, name, voice_dir)
            row["i"] = i
            print(json.dumps(row, ensure_ascii=False), file=out, flush=True)


if __name__ == "__main__":
    main()
