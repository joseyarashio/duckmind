/* Where the duck on screen comes from (phase 0 「看見真相」, spec §13 and §17).
 *
 * With live physics the solid duck is the physics truth only: where the body is in the world
 * (the frame's `truth`: x_mm, y_mm, yaw_mrad in the home's world coordinates — phase 2; in
 * default_home the same as Duck Core's room — and `placed_at`, the last time a world operation
 * moved the body), its height, lean and joints (the frame's trunk z, quat and joints). Duck
 * Core's plan — its own idea of where it is and what it does, moved into the world by the
 * home's frame — moves only the shadow duck. Without truth (a fast-forward, no live physics, a
 * server that sends none) the solid duck is the plan, and the page says so.
 *
 * Pure functions: no three.js, no DOM. duck3d.js draws with them; tests run them in node. */

export const HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
export const TRAIL_MS = 10 * 60 * 1000; // the truth trail: the last 10 minutes
export const TRAIL_MAX = 9000; // points kept (a walk adds one about every 1–2 cm)
const TRAIL_STEP_M = 0.01;
const TRAIL_GAP_M = 0.5; // farther apart than this: a break in the line, never a jump drawn
const SNAP_M = 0.5;

// Words are catalog keys (static/i18n.js renders them in the viewer's language; 繁體中文
// 「物理真相」 and 「這是小鴨的計畫，不是物理」).
export const TRUTH_TITLE = "page.truth.TRUTH_TITLE";
export const PLAN_TITLE = "page.truth.PLAN_TITLE";

// The heading (about the vertical) of a MuJoCo quaternion [w, x, y, z].
export function yawOf(q) {
  const [w, x, y, z] = q;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function quatMul(a, b) {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

// The same orientation (its lean) turned to face `yaw`.
export function withHeading(quat, yaw) {
  const own = yawOf(quat);
  const tilt = quatMul([Math.cos(-own / 2), 0, 0, Math.sin(-own / 2)], quat);
  return quatMul([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], tilt);
}

// What one physics frame says about the duck in the world, or null when it carries no truth.
// Reads the frame only: its truth (position, heading, placement), trunk height, quat, joints.
export function truthTarget(frame) {
  const t = frame && frame.truth;
  if (!t || t.x_mm == null || t.y_mm == null || t.yaw_mrad == null) return null;
  if (!Array.isArray(frame.trunk) || !Array.isArray(frame.quat)) return null;
  const yaw = t.yaw_mrad / 1000;
  const norm = Math.hypot(...frame.quat) || 1;
  return {
    x: t.x_mm / 1000,
    y: t.y_mm / 1000,
    yaw,
    z: frame.trunk[2],
    quat: withHeading(frame.quat.map((v) => v / norm), yaw),
    joints: { ...(frame.joints || {}) },
    placedAt: t.placed_at ?? null,
    why: t.placed_why || "",
  };
}

function angleTo(from, to, k) {
  const d = Math.atan2(Math.sin(to - from), Math.cos(to - from));
  return from + d * k;
}

function nlerp(a, b, k) {
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = dot < 0 ? -1 : 1;
  const q = a.map((v, i) => v + (s * b[i] - v) * k);
  const n = Math.hypot(...q) || 1;
  return q.map((v) => v / n);
}

// Frames arrive ~16 times a second: ease toward the newest so motion stays smooth. A world
// placement (the body started, a sync after a fast-forward: x_place moved it) or a jump is
// shown as it is, at once: the drawn duck never slides across the room to get there. A hand
// carrying the body is no jump: it is followed all the way.
export function follow(shown, target, dt) {
  if (!target) return null;
  const snap =
    !shown || shown.placedAt !== target.placedAt || Math.hypot(target.x - shown.x, target.y - shown.y) > SNAP_M;
  if (snap) return { ...target, joints: { ...target.joints }, quat: [...target.quat] };
  const k = Math.min(1, dt * 18);
  const joints = { ...shown.joints };
  for (const [name, value] of Object.entries(target.joints)) {
    const old = joints[name] ?? value;
    joints[name] = old + (value - old) * k;
  }
  return {
    ...target,
    x: shown.x + (target.x - shown.x) * k,
    y: shown.y + (target.y - shown.y) * k,
    z: shown.z + (target.z - shown.z) * k,
    yaw: angleTo(shown.yaw, target.yaw, k),
    quat: nlerp(shown.quat, target.quat, k),
    joints,
  };
}

/* ---- every physics sample, at its own moment (fid:page 2026-09-28) ----
 * The server keeps a ring of the body's samples (/api/body?since_t=…: `frames`, 50 a second in
 * the prototype) and the page plays each one at the moment it happened, a short display delay
 * behind, interpolating only between two true samples. It never eases toward the newest frame:
 * a 0.3 s head snap is drawn in 0.3 s at its full size (`follow` above is only for a server
 * without frames). Times are the server's wall clock (`w`, epoch seconds of the read), so the
 * beak and the calls (their wall_ms) play on the same clock as the joints. */
export const FRAMES_KEEP_S = 3; // samples kept on the page
export const FRAME_GAP_S = 0.25; // two samples farther apart than this: hold, never glide across
export const PLAY_MIN_S = 0.06; // the display delay's bounds
export const PLAY_MAX_S = 0.8;
const PLAY_SLEW = 0.04; // the page's clock follows its targets at most 4 % of real time
const OFFSET_WINDOW_S = 10; // the clocks' offset: the quickest answer of the last 10 s
const STALE_WINDOW_S = 6; // the delay: how stale the newest sample got, over the last 6 s

// One compact frame (livebody._keep_frame) with the server's joint names.
export function unpackFrame(frame, names) {
  const joints = {};
  (names || []).forEach((name, i) => {
    const value = frame.j ? frame.j[i] : undefined;
    if (value != null) joints[name] = value;
  });
  const mouth = Number.isFinite(frame.m) ? frame.m : null;
  const knownSource = ["measured_sim", "commanded_fallback", "unavailable"].includes(frame.m_source)
    ? frame.m_source
    : "unavailable";
  return { t: frame.t, w: frame.w, joints, trunk: [...frame.p], quat: [...frame.q], mouth, mouthSource: mouth == null ? "unavailable" : knownSource, calls: frame.v || null };
}

const BEAK_READOUT_KEYS = {
  measured_sim: "page.duck3d.readout.beak.measured",
  commanded_fallback: "page.duck3d.readout.beak.commanded",
  unavailable: "page.duck3d.readout.beak.unavailable",
};

export function beakCaption(i18n, source, value) {
  const key = BEAK_READOUT_KEYS[source] || BEAK_READOUT_KEYS.unavailable;
  if (!i18n || typeof i18n.t !== "function") return "";
  return source === "measured_sim" || source === "commanded_fallback"
    ? i18n.t(key, { value })
    : i18n.t(key);
}

// The calls on the samples the playing passed since the last draw (after `fromW`, up to `toW`):
// each sounds with the sample that carried it.
export function callsBetween(buffer, fromW, toW) {
  const out = [];
  if (fromW == null || toW == null || toW <= fromW) return out;
  for (const f of buffer) {
    if (f.w > fromW && f.w <= toW && f.calls) out.push(...f.calls);
  }
  return out;
}

// New frames into the page's buffer (sorted by time, each sample once); a body whose clock
// started over (a restart) replaces the buffer. Returns the buffer.
export function mergeFrames(buffer, incoming, keepS = FRAMES_KEEP_S) {
  let out = buffer;
  for (const f of incoming) {
    const last = out[out.length - 1];
    if (last && f.t <= last.t) {
      if (f.t < last.t - 1) out = [f]; // (a new body: its clock starts over)
      continue;
    }
    out.push(f);
  }
  const newest = out.length ? out[out.length - 1].w : 0;
  let drop = 0;
  while (drop < out.length - 2 && newest - out[drop].w > keepS) drop++;
  if (drop) out.splice(0, drop);
  return out;
}

function lerpPose(a, b, u) {
  const joints = {};
  for (const [name, value] of Object.entries(a.joints)) {
    const next = b.joints[name];
    joints[name] = next == null ? value : value + (next - value) * u;
  }
  const quat = nlerp(a.quat, b.quat, u);
  const trunk = a.trunk.map((v, i) => v + (b.trunk[i] - v) * u);
  const mouth = a.mouth == null ? null : b.mouth == null ? a.mouth : a.mouth + (b.mouth - a.mouth) * u;
  return { t: a.t + (b.t - a.t) * u, w: a.w + (b.w - a.w) * u, joints, trunk, quat, mouth, mouthSource: conservativeMouthSource(a.mouthSource, b.mouthSource, mouth) };
}

// Never describe an interpolated command or unknown sample as a measured beak position.
function conservativeMouthSource(a, b, mouth) {
  if (!Number.isFinite(mouth)) return "unavailable";
  if (a === b && ["measured_sim", "commanded_fallback", "unavailable"].includes(a)) return a;
  if (["measured_sim", "commanded_fallback"].includes(a) && ["measured_sim", "commanded_fallback"].includes(b)) {
    return "commanded_fallback";
  }
  return "unavailable";
}

// The body at server wall time `atW`: between the two true samples around it (linear joints and
// position, normalised quaternion), or the nearest sample held (before the first, after the
// newest — never guessed ahead — or across a gap or a jump). `state`: "between" | "hold" |
// "late" (the newest is older than `atW`) | "empty".
export function sampleFrames(buffer, atW) {
  if (!buffer.length) return { pose: null, state: "empty" };
  const last = buffer[buffer.length - 1];
  if (atW >= last.w) return { pose: last, state: "late", behind: atW - last.w };
  if (atW <= buffer[0].w) return { pose: buffer[0], state: "hold" };
  let lo = 0;
  let hi = buffer.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (buffer[mid].w <= atW) lo = mid;
    else hi = mid;
  }
  const a = buffer[lo];
  const b = buffer[hi];
  const jump = Math.hypot(b.trunk[0] - a.trunk[0], b.trunk[1] - a.trunk[1]) > SNAP_M;
  if (b.t - a.t > FRAME_GAP_S || b.w - a.w > FRAME_GAP_S || jump) return { pose: a, state: "hold" };
  const u = (atW - a.w) / Math.max(1e-6, b.w - a.w);
  return { pose: lerpPose(a, b, Math.min(1, Math.max(0, u))), state: "between" };
}

// What the view draws from a played sample (the shape truthTarget gives for a whole frame).
export function framePose(sample, placedAt = null) {
  const norm = Math.hypot(...sample.quat) || 1;
  const quat = sample.quat.map((v) => v / norm);
  return {
    x: sample.trunk[0],
    y: sample.trunk[1],
    yaw: yawOf(quat),
    z: sample.trunk[2],
    quat,
    joints: { ...sample.joints },
    placedAt,
    why: "",
    mouth: sample.mouth,
    mouthSource: Number.isFinite(sample.mouth) && ["measured_sim", "commanded_fallback"].includes(sample.mouthSource)
      ? sample.mouthSource
      : "unavailable",
    t: sample.t,
    w: sample.w,
  };
}

// The page's clock for the server's samples. `off`: the server's wall clock minus the page's
// (the quickest answer of the last few seconds: the least delayed); `delay`: how far behind that
// the page plays — the 90th percentile of how stale its newest sample had become just before
// each answer (network, the polling gaps), plus a little. Both move toward their targets slowly,
// so the playing never jumps (no motion replayed or skipped).
export function makePlayout() {
  return { offs: [], stale: [], off: null, offTarget: null, delay: 0.15, target: 0.15, newestW: null };
}

function pruneWindow(items, nowS, windowS, max) {
  while (items.length > 2 && nowS - items[0].at > windowS) items.shift();
  if (items.length > max) items.splice(0, items.length - max);
}

export function playoutArrive(play, newestW, nowS) {
  if (newestW == null || !Number.isFinite(newestW)) return play;
  if (play.newestW != null && newestW < play.newestW - 1) {
    // (the server's clock went back: a restarted server or another machine; start over)
    play.offs = [];
    play.stale = [];
    play.off = null;
    play.newestW = null;
  }
  if (play.newestW != null && play.off != null) {
    play.stale.push({ at: nowS, v: nowS + play.off - play.newestW });
    pruneWindow(play.stale, nowS, STALE_WINDOW_S, 120);
    const sorted = play.stale.map((s) => s.v).sort((x, y) => x - y);
    const p90 = sorted[Math.min(sorted.length - 1, Math.floor(0.9 * sorted.length))];
    play.target = Math.min(PLAY_MAX_S, Math.max(PLAY_MIN_S, p90 + 0.015));
  }
  play.offs.push({ at: nowS, v: newestW - nowS });
  pruneWindow(play.offs, nowS, OFFSET_WINDOW_S, 400);
  play.offTarget = Math.max(...play.offs.map((o) => o.v));
  if (play.off == null) play.off = play.offTarget;
  play.newestW = play.newestW == null ? newestW : Math.max(play.newestW, newestW);
  return play;
}

// The server wall time to draw now (`nowS`: the page's clock in seconds; `dt` since last call).
export function playoutTime(play, nowS, dt) {
  if (play.off == null) return null;
  const step = PLAY_SLEW * Math.max(0, Math.min(dt, 0.25));
  const toward = (value, goal) => value + Math.max(-step, Math.min(step, goal - value));
  play.off = toward(play.off, play.offTarget ?? play.off);
  play.delay = toward(play.delay, play.target);
  return nowS + play.off - play.delay;
}

// The two ducks the view draws. `plan`: Duck Core's pose as the view animates it
// ({joints, trunk, quat, mouth}); `truth`: the followed truth (or null). With truth, the
// solid duck's position, heading, height, lean and legs are the physics body's; its head is
// the physics head too unless `physicsHead` is off (then Duck Core's command); the mouth is
// the command (the physics model has no mouth). The plan moves the shadow duck only.
export function compose({ plan, truth, physicsHead = true }) {
  if (!truth) return { mode: "plan", solid: plan, shadow: null };
  const joints = { ...truth.joints };
  if (!physicsHead) {
    for (const name of HEAD) if (plan.joints && plan.joints[name] != null) joints[name] = plan.joints[name];
  }
  return {
    mode: "truth",
    solid: { joints, trunk: [truth.x, truth.y, truth.z], quat: [...truth.quat], mouth: plan.mouth },
    shadow: plan,
  };
}

// The truth trail: one point per centimetre moved (or every few seconds standing still),
// the last 10 minutes of wall-clock time, broken where a world placement moved the body.
export function trailPush(trail, point, now) {
  while (trail.length && now - trail[0].t > TRAIL_MS) trail.shift();
  const last = trail[trail.length - 1];
  if (last && last.a === point.a && now - last.t < 4000 && Math.hypot(point.x - last.x, point.y - last.y) < TRAIL_STEP_M) return false;
  trail.push({ x: point.x, y: point.y, a: point.a, t: now });
  if (trail.length > TRAIL_MAX) trail.splice(0, trail.length - TRAIL_MAX);
  return true;
}

// The trunk sways sideways with every step: the line drawn is the path, each point averaged
// with its neighbours within ± this many milliseconds (about one gait period), same placement.
export const TRAIL_SMOOTH_MS = 300;

export function smoothTrail(trail, windowMs = TRAIL_SMOOTH_MS) {
  return trail.map((p, i) => {
    let sx = 0;
    let sy = 0;
    let n = 0;
    for (let j = i; j >= 0 && p.t - trail[j].t <= windowMs && trail[j].a === p.a; j--) {
      sx += trail[j].x;
      sy += trail[j].y;
      n += 1;
    }
    for (let j = i + 1; j < trail.length && trail[j].t - p.t <= windowMs && trail[j].a === p.a; j++) {
      sx += trail[j].x;
      sy += trail[j].y;
      n += 1;
    }
    return { ...p, x: sx / n, y: sy / n };
  });
}

// The trail's line pieces [x0, y0, x1, y1] (room metres), without the pieces across a break.
export function trailSegments(trail, now, windowMs = TRAIL_SMOOTH_MS) {
  const path = windowMs > 0 ? smoothTrail(trail, windowMs) : trail;
  const out = [];
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1];
    const b = path[i];
    if (now - a.t > TRAIL_MS || a.a !== b.a || Math.hypot(b.x - a.x, b.y - a.y) > TRAIL_GAP_M) continue;
    out.push([a.x, a.y, b.x, b.y]);
  }
  return out;
}

// How different two drawn poses look (metres, roughly): apart on the floor, in height (sitting
// against standing, lying down), and turned or tipped. The shadow duck fades out where it
// would only lie on top of the solid duck.
export function poseGap(a, b) {
  if (!a || !b) return Infinity;
  const dot = Math.abs(a.quat[0] * b.quat[0] + a.quat[1] * b.quat[1] + a.quat[2] * b.quat[2] + a.quat[3] * b.quat[3]);
  const angle = 2 * Math.acos(Math.min(1, dot));
  return Math.hypot(a.trunk[0] - b.trunk[0], a.trunk[1] - b.trunk[1]) + 2 * Math.abs(a.trunk[2] - b.trunk[2]) + 0.1 * angle;
}

// The words over the 3D view: which duck is drawn, and why that one. `broken`: the physics
// body has an error (its plain reason is in 「物理不可信」); `oldServer`: the server sends
// frames without the truth this page reads (an older version); `physicalHome`: the physics
// has the drawn home's walls and furniture (phase 2) — else the translucent room is only on
// the page; `held`: the world's hand holds the body up. The title and detail are catalog keys.
export function modeLabel({ truth, enabled, on, available, fastForward, offline, demo, planFallen, physicsFallen, broken, oldServer, physicalHome, held }) {
  if (truth) {
    let detail = physicalHome ? "page.truth.modeLabel.a501b2" : "page.truth.modeLabel.ad8793";
    if (demo) detail = "page.truth.modeLabel.403afa";
    else if (held) detail = "page.truth.modeLabel.86c49d";
    else if (physicsFallen) detail = "page.truth.modeLabel.7286ee";
    else if (planFallen) detail = "page.truth.modeLabel.7478ab";
    return { mode: "truth", title: TRUTH_TITLE, detail };
  }
  let detail = "page.truth.modeLabel.c24ec4";
  if (offline) detail = "page.truth.modeLabel.ee7586";
  else if (enabled && !on) detail = "page.truth.modeLabel.72762e";
  else if (enabled && broken) detail = "page.truth.modeLabel.f81b06";
  else if (enabled && fastForward) detail = "page.truth.modeLabel.f2b6b4";
  else if (enabled && !available) detail = "page.truth.modeLabel.d97b0c";
  else if (enabled && oldServer) detail = "page.truth.modeLabel.1a28ba";
  else if (enabled) detail = "page.truth.modeLabel.d8be5e";
  return { mode: "plan", title: PLAN_TITLE, detail };
}
