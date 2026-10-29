"""fid:page (2026-09-28): the 3D view draws every physics sample at its true speed.

The owner judges only what the body visibly does in the 3D view. Before this, the page asked
/api/body ~14 times a second and eased the drawn duck toward the newest frame (``follow``: a
55 ms first-order lag on top of ~70 ms polling): a 0.3 s head snap was drawn late, slower and
smaller. Now the server keeps a ring of body samples (50 a second in the prototype) and the page
plays each one at its moment, a short display delay behind, interpolating only between two true
samples. These tests run ``static/truth.js`` in node (no browser, no physics) and the server's
frame ring with a stand-in body.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from duckmind import livebody
from duckmind.livebody import FRAME_JOINTS, LiveBody, LivePaths

STATIC = Path(__file__).resolve().parents[1] / "src" / "duckmind" / "simweb" / "static"
TRUTH_JS = STATIC / "truth.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(script: str) -> Any:
    source = f'import * as T from "{TRUTH_JS.as_uri()}";\n{script}'
    done = subprocess.run(
        [str(NODE), "--input-type=module", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# A head snap as the physics has it: head_pitch 0 → −0.35 rad (20°) in 0.3 s (smooth), held,
# then back; sampled at 50 Hz on the server's wall clock (w), a small read jitter.
SNAP = """
const names = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
const truthAt = (w) => {
  const u = Math.min(1, Math.max(0, (w - 10.0) / 0.3));
  const back = Math.min(1, Math.max(0, (w - 11.0) / 0.3));
  const s = (x) => x * x * (3 - 2 * x);
  return -0.35 * (s(u) - s(back));
};
const frames = [];
let seed = 7;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
for (let i = 0; i < 150; i++) {
  const w = 9.5 + i * 0.02 + (rnd() - 0.5) * 0.004;
  frames.push({ t: i * 0.02, w, j: [0.35, 0.35 + truthAt(w), 0, 0], p: [1, 2, 0.12], q: [1, 0, 0, 0] });
}
// The page: answers every 60–90 ms (the poll), each with the samples read since the last one,
// arriving 5–40 ms after the newest was read; drawing at 60 frames a second.
const play = T.makePlayout();
let buffer = [];
let next = 0;
let answerAt = 9.55;
const drawn = [];
let late = 0;
let last = null;
for (let now = 9.5; now < 12.4; now += 1 / 60) {
  while (answerAt <= now) {
    const upTo = answerAt - 0.005 - rnd() * 0.035;
    const fresh = [];
    while (next < frames.length && frames[next].w <= upTo) fresh.push(frames[next++]);
    if (fresh.length) {
      buffer = T.mergeFrames(buffer, fresh.map((f) => T.unpackFrame(f, names)));
      T.playoutArrive(play, buffer[buffer.length - 1].w, answerAt);
    }
    answerAt += 0.06 + rnd() * 0.03;
  }
  const at = T.playoutTime(play, now, last == null ? 0 : now - last);
  last = now;
  if (at == null) continue;
  const got = T.sampleFrames(buffer, at);
  if (got.state === "late") late++;
  if (got.pose) drawn.push({ now, at, v: got.pose.joints.head_pitch - 0.35, truth: truthAt(at) });
}
"""


@needs_node
def test_a_head_snap_is_drawn_at_its_true_speed_and_size():
    out = _node(
        SNAP
        + """
const snap = drawn.filter((d) => d.at >= 9.9 && d.at <= 10.8);
const peak = Math.min(...snap.map((d) => d.v));
// when the drawn head passed 10 % and 90 % of the snap, on the server's clock
const cross = (frac) => snap.find((d) => d.v <= -0.35 * frac).at;
const err = Math.max(...drawn.filter((d) => d.at > 9.7).map((d) => Math.abs(d.v - d.truth)));
// how fast the page's clock ran against the server's while it played (1 = true speed)
const mid = drawn.filter((d) => d.now > 10.5);
const rate = (mid[mid.length - 1].at - mid[0].at) / (mid[mid.length - 1].now - mid[0].now);
console.log(JSON.stringify({ peak, rise: cross(0.9) - cross(0.1), err, late, n: drawn.length,
  delay: play.delay, rate }));
"""
    )
    # the full 20° reached (never eased smaller), in the snap's own time (0.3 s: 10 %→90 % of a
    # smoothstep is 0.19 s), drawn within 1° of the physics at every frame
    assert out["peak"] == pytest.approx(-0.35, abs=0.004)
    assert out["rise"] == pytest.approx(0.19, abs=0.025)
    assert out["err"] < 0.017  # rad (1°): linear between 20 ms samples of a 0.3 s snap
    # the page's clock runs at the server's pace, a little behind (the display delay)
    assert out["rate"] == pytest.approx(1.0, abs=0.03)
    assert 0.06 <= out["delay"] <= 0.25
    assert out["late"] <= 0.03 * out["n"]  # it (almost) never waits for a sample


FLINCH = """
const names = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
// a flinch: the head out 20° and back within 0.25 s, sampled at 50 Hz
const pulse = (w) => { const u = (w - 10.0) / 0.25; return u > 0 && u < 1 ? -0.35 * Math.sin(Math.PI * u) : 0; };
const fr = [];
for (let i = 0; i < 150; i++) {
  const w = 9.5 + i * 0.02;
  fr.push({ t: i * 0.02, w, j: [0.35, 0.35 + pulse(w), 0, 0], p: [1, 2, 0.12], q: [1, 0, 0, 0] });
}
// before: the newest frame asked every `poll` s and eased (follow)
const before = (poll) => {
  let shown = null, polled = null, pollAt = 9.5, peak = 0, peakAt = 0;
  for (let now = 9.5; now < 11; now += 1 / 60) {
    if (now >= pollAt) {
      const n = fr.filter((f) => f.w <= now - 0.02).pop();
      if (n) polled = { truth: { x_mm: 1000, y_mm: 2000, yaw_mrad: 0, placed_at: 1 }, trunk: n.p, quat: n.q,
        joints: { neck_pitch: n.j[0], head_pitch: n.j[1], head_yaw: 0, head_roll: 0 } };
      pollAt = now + poll;
    }
    if (!polled) continue;
    shown = T.follow(shown, T.truthTarget(polled), 1 / 60);
    const v = shown.joints.head_pitch - 0.35;
    if (v < peak) { peak = v; peakAt = now; }
  }
  return { peak: peak * 180 / Math.PI, late: peakAt - 10.125 };
};
// now: every sample, answers every `poll` s, played on the page's clock
const after = (poll) => {
  const play = T.makePlayout();
  let buf = [], nx = 0, ans = 9.55, last = null, peak = 0;
  for (let now = 9.5; now < 11; now += 1 / 60) {
    while (ans <= now) {
      const fresh = [];
      while (nx < fr.length && fr[nx].w <= ans - 0.02) fresh.push(fr[nx++]);
      if (fresh.length) {
        buf = T.mergeFrames(buf, fresh.map((f) => T.unpackFrame(f, names)));
        T.playoutArrive(play, buf[buf.length - 1].w, ans);
      }
      ans += poll;
    }
    const at = T.playoutTime(play, now, last == null ? 0 : now - last);
    last = now;
    if (at == null) continue;
    const g = T.sampleFrames(buf, at);
    if (g.pose) peak = Math.min(peak, g.pose.joints.head_pitch - 0.35);
  }
  return { peak: peak * 180 / Math.PI, delay: play.delay };
};
"""


@needs_node
def test_a_short_flinch_was_drawn_smaller_and_late_and_now_is_drawn_whole():
    """The before: the page asked for the newest frame every ~70 ms (localhost; ~110 ms through
    the tailnet) and eased toward it — a 0.25 s flinch of 20° was drawn at 18° (17° through the
    tailnet), its top 0.11 s (0.14 s) late. Now: 20° at every sample, a fixed ~0.1 s display
    delay (the time an answer takes to arrive)."""
    out = _node(
        FLINCH
        + """
console.log(JSON.stringify({ local: before(0.07), tailnet: before(0.11), now: after(0.075), nowTailnet: after(0.11) }));
"""
    )
    assert out["local"]["peak"] > -18.6 and out["local"]["late"] > 0.08  # smaller and late
    assert out["tailnet"]["peak"] > out["local"]["peak"]  # worse with slower answers
    for key in ("now", "nowTailnet"):
        assert out[key]["peak"] == pytest.approx(-20.0, abs=0.2)  # the whole flinch
    assert out["now"]["delay"] <= 0.15 and out["nowTailnet"]["delay"] <= 0.2


@needs_node
def test_frames_merge_once_restart_cleanly_and_calls_sound_once():
    out = _node(
        """
const names = ["head_pitch"];
const f = (t, w, v) => ({ t, w, j: [0.1], p: [0, 0, 0.1], q: [1, 0, 0, 0], ...(v ? { v } : {}) });
let buf = T.mergeFrames([], [f(1, 101), f(1.02, 101.02, [{ tag: "chirp", level: 0.5 }]), f(1.04, 101.04)].map((x) => T.unpackFrame(x, names)));
buf = T.mergeFrames(buf, [f(1.02, 101.02), f(1.04, 101.04), f(1.06, 101.06)].map((x) => T.unpackFrame(x, names)));
const once = buf.length;
const calls1 = T.callsBetween(buf, 101.0, 101.03).length;
const calls2 = T.callsBetween(buf, 101.03, 101.06).length;
const restarted = T.mergeFrames(buf, [f(0.02, 200)].map((x) => T.unpackFrame(x, names)));
// a gap (the physics stalled): held, never glided across
const gap = T.mergeFrames([], [f(1, 1), f(1.6, 1.6)].map((x) => T.unpackFrame(x, names)));
const held = T.sampleFrames(gap, 1.3);
console.log(JSON.stringify({ once, calls1, calls2, restarted: restarted.length, held: held.state, heldW: held.pose.w }));
"""
    )
    assert out["once"] == 4  # each sample once
    assert (out["calls1"], out["calls2"]) == (1, 0)  # a call sounds with its sample, once
    assert out["restarted"] == 1  # a new body's clock: the old samples are gone
    assert out["held"] == "hold" and out["heldW"] == 1


# ── the server's frame ring (a stand-in body: no robotd, no MuJoCo) ─────────────


class _Oracle:
    def __init__(self) -> None:
        self.t = 0.0

    def snapshot(self) -> dict[str, Any]:
        self.t += 0.02
        return {
            "positions": [0.01 * i for i in range(15)],
            "sim_time": self.t,
            "trunk": [1.0, 2.0, 0.12],
            "trunk_z": 0.12,
            "imu": {"quat": [1.0, 0.0, 0.0, 0.0]},
        }


def _body() -> LiveBody:
    body = LiveBody(
        LivePaths(
            Path("/nowhere"), Path("/nowhere"), Path("/nowhere"), Path("/nowhere"), Path("/nowhere")
        )
    )
    body._oracle = _Oracle()  # type: ignore[assignment]
    return body


def test_the_body_keeps_every_sample_for_the_page_with_the_beak_and_the_calls():
    body = _body()
    body.mouth_cmd = 0.4
    for i in range(10):
        if i == 3:
            body.voice("greet", 0.8)
        snap = body._oracle.snapshot()  # type: ignore[union-attr]
        latest = {
            "t": snap["sim_time"],
            "joints": dict(zip(FRAME_JOINTS, [0.1] * len(FRAME_JOINTS), strict=True)),
            "trunk": snap["trunk"],
            "quat": snap["imu"]["quat"],
        }
        body._keep_frame(latest, 1000.0 + i * 0.02)
        body._keep_frame(latest, 1000.0 + i * 0.02 + 0.001)  # (the same sample read twice)
    frames = body.frames_since(None)
    assert len(frames) == 10  # each once
    assert [len(f["j"]) for f in frames] == [len(FRAME_JOINTS)] * 10
    assert all(f["m"] == 0.4 for f in frames)
    assert [f.get("v") for f in frames].count(None) == 9
    assert frames[3]["v"] == [{"tag": "greet", "level": 0.8}]
    # since the page's newest: only the newer ones; a time from another body: all of them
    assert len(body.frames_since(frames[6]["t"])) == 3
    assert len(body.frames_since(999.0)) == 10
    # a restarted body's clock starts over: the ring starts over with it
    body._keep_frame({**latest, "t": 0.02}, 2000.0)
    assert [f["t"] for f in body.frames_since(None)] == [0.02]


def test_prototype_mode_reads_the_body_every_control_tick():
    from duckmind.simweb import live

    assert live.NEURO_SAMPLE_HZ == 60 and livebody.SAMPLE_HZ == 30
    assert "body.sample_hz = float(NEURO_SAMPLE_HZ)" in Path(live.__file__).read_text()
